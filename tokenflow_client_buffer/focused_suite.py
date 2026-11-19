"""User-requested 32-sequence, two-arm, two-repeat experiment controller.

Can take over a fully drained text service without restarting its model. The
inference/client source snapshot is unchanged; only the experiment queue changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import urllib.request

from tokenflow_client_buffer.native_suite import CONTAINER, Suite, save


class FocusedSuite(Suite):
    def progress(self, status, **extra):
        save(self.root / "progress.json", dict(
            status=status, updated_at=time.time(), controller_pid=os.getpid(),
            cell=self.cell, completed_formal_cases=len(self.completed),
            completed_cases=self.completed, planned_formal_cases=8, **extra))

    def adopt_text(self):
        manifest = json.loads((self.root / "runtime/active_launch.json").read_text())
        config = manifest["scheduler_config"]
        if (manifest["max_num_seqs"] != 32 or not manifest["adapter_enabled"]
                or config["modality"] != "text" or "execution_cap" in config):
            raise RuntimeError("existing service does not match requested text configuration")
        trace = Path(manifest["trace"])
        if trace.parent != self.root / "runtime":
            raise RuntimeError("existing trace belongs to another experiment")
        info = json.loads(subprocess.check_output(
            ["docker", "inspect", CONTAINER], text=True))[0]
        if not info["State"]["Running"] or not any(
                m["Source"] == str(self.root / "framework")
                and m["Destination"] == str(self.root / "framework")
                for m in info["Mounts"]):
            raise RuntimeError("existing container does not belong to this experiment")
        with urllib.request.urlopen("http://127.0.0.1:18195/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("existing service is unhealthy")
        self.running_container = True
        self.launch_index = 1
        return trace

    def case(self, name, modality, arm, workload, trace, profile=None, formal=False):
        from tokenflow_client_buffer.verify_run import verify
        case = self.root / "runs" / name
        if (case / "complete.json").exists():
            evidence = verify(case, trace)
            rows = json.loads((case / "client/results.json").read_text())
            if (evidence["errors"] or evidence["unresolved_pauses"]
                    or any(r["startup_s"] != r["first_ready_s"] for r in rows)):
                raise RuntimeError("existing case failed validation: " + name)
        else:
            case = self.run_case(name, modality, arm, workload, trace, profile)
        if formal:
            self.completed.append(name)
            self.progress("case_complete", case=name)
        return case

    def run_focused(self, adopt_text=False):
        from tokenflow_client_buffer.report import aggregate, plots
        manifest = json.loads((self.root / "runtime/source_manifest.json").read_text())
        for name, digest in manifest.items():
            if hashlib.sha256((self.root / "framework" / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("frozen source changed: " + name)

        def hashes(case):
            return {r["request_id"]: r["content_sha256"] for r in
                    json.loads((case / "client/results.json").read_text())}

        try:
            for modality in ("text", "audio"):
                cap = 32
                self.cell = f"{modality}_{cap}"
                self.progress("adopting_service" if modality == "text" and adopt_text
                              else "preparing_service")
                trace = (self.adopt_text() if modality == "text" and adopt_text
                         else self.start(modality, cap, True))
                warm = self.small_workload(modality, cap, "warmup", 4)
                self.case(f"{self.cell}_shared_warmup", modality, "fcfs", warm, trace)
                calibration = self.small_workload(modality, cap, "calibration", cap * 2)
                case = self.case(f"{self.cell}_calibration", modality, "fcfs", calibration, trace)
                profile = self.root / "workloads" / f"{self.cell}_capacity.json"
                if not profile.exists():
                    stats = json.loads((case / "client/summary.json").read_text())
                    save(profile, dict(modality=modality, raw_capacity=stats["generated_units_per_s"],
                        units=stats["units"], duration_s=stats["generated_window_s"], max_num_seqs=cap,
                        source_case=str(case), scope="independent native-policy calibration; same six GPUs and bounded workload"))
                smoke = self.small_workload(modality, cap, "smoke", cap + 8)
                native = self.case(f"{self.cell}_native_smoke", modality, "fcfs", smoke, trace)
                buffer = self.case(f"{self.cell}_buffer_smoke", modality, "buffer", smoke, trace, profile)
                if hashes(native) != hashes(buffer):
                    raise RuntimeError("shared-instance native/buffer smoke output mismatch")
                for rep in range(2):
                    arms = ["fcfs", "buffer"]
                    random.Random(92800 + cap + rep).shuffle(arms)
                    paired = {}
                    for arm in arms:
                        paired[arm] = self.case(f"{modality}_c{cap}_formal_{rep}_{arm}", modality, arm,
                            self.root / "workloads" / f"{modality}_{rep}.jsonl", trace,
                            None if arm == "fcfs" else profile, formal=True)
                    if hashes(paired["fcfs"]) != hashes(paired["buffer"]):
                        raise RuntimeError(f"formal content mismatch: {modality}/{cap}/{rep}")
                self.stop()
                prefix = f"{modality}_c{cap}_formal"
                summary, pairs = aggregate(self.root, prefix, 2)
                plots(self.root, prefix, summary, pairs)
            self.progress("complete")
        except BaseException as error:
            self.progress("stopped_with_error", error=repr(error))
            raise
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--adopt-text", action="store_true")
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    FocusedSuite(args.root.resolve()).run_focused(args.adopt_text)


if __name__ == "__main__":
    main()
