"""Prepare/run the six-GPU native-concurrency comparison; never stop other jobs.

Preparation is CPU-only. Run mode waits for the original GPUs 0--5 to be idle,
then runs pure-text and pure-audio experiments sequentially. A failed smoke,
case, or content/evidence check stops the suite and releases its own container.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request


IMAGE = "exp1/vllm-omni-stage01:20261028-6840e2c9-pandas"
CONTAINER = "zy-tokenflow-native-20261109"
MODEL = "/data/models/Qwen3-Omni-30B-A3B-Instruct"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def prepare(root, previous, repo):
    if (root / "experiment.json").exists():
        raise ValueError("experiment already prepared; do not overwrite frozen inputs")
    for name in ("runtime", "runs", "workloads", "reports", "validation"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(previous / "framework", root / "framework",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source in (repo / "tokenflow_client_buffer").glob("*.py"):
        shutil.copy2(source, root / "framework/tokenflow_client_buffer" / source.name)
    for name in ("pyproject.toml", "vllm_omni/core/sched/omni_ar_scheduler.py",
                 "vllm_omni/profiling/stage_queue_trace.py"):
        shutil.copy2(repo / name, root / "framework" / name)
    manifest = {}
    for path in (root / "framework").rglob("*.py"):
        manifest[str(path.relative_to(root / "framework"))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(root / "runtime/source_manifest.json", manifest)
    workloads = {}
    for modality in ("text", "audio"):
        name = "text_formal_0.jsonl" if modality == "text" else "audio_formal_0_bounded.jsonl"
        templates = [json.loads(line) for line in (previous / "runs" / name).read_text().splitlines()]
        for rep in range(3):
            order = list(range(200))
            random.Random(91300 + rep).shuffle(order)
            rows = []
            for index, prompt_index in enumerate(order):
                row = copy.deepcopy(templates[prompt_index % len(templates)])
                row.pop("startup_s", None)
                row["arrival_s"] = index * .005 if index < 50 else 1.5 + (index - 50) * .005
                meta = row["payload"]["omni_scheduling"]
                meta.update(source_request_id=f"{modality}-native-{rep}-{index}", ingress_order=index)
                if modality == "text":
                    meta["text_consume_rate_tps"] = [5., 10., 15.][prompt_index % 3]
                rows.append(row)
            path = root / "workloads" / f"{modality}_{rep}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            workloads[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cap in (32, 64):
            raw = (previous / "runtime" / ("deploy.yaml" if modality == "text" else "deploy_audio.yaml")).read_text()
            raw = raw.replace(str(previous), str(root)).replace("max_num_seqs: 16", f"max_num_seqs: {cap}")
            (root / "runtime" / f"deploy_{modality}_{cap}.yaml").write_text(raw)
            cfg = json.loads((previous / "runtime" / ("scheduler.json" if modality == "text" else "scheduler_audio.json")).read_text().replace(str(previous), str(root)))
            cfg.pop("execution_cap")
            save(root / "runtime" / f"scheduler_{modality}_{cap}.json", cfg)
    save(root / "experiment.json", dict(
        model=MODEL, container=CONTAINER, image=IMAGE, physical_gpus=list(range(6)),
        thinker_tp=4, talker_gpus=1, code2wav_gpus=1, max_num_seqs=[32, 64],
        requests_per_case=200, repetitions=3, modalities=["text", "audio"],
        arms=["fcfs", "capacity_only", "buffer"], total_formal_cases=36,
        workload_sha256=workloads, playback_start_policy="first_ready",
        comparison="All arms share one model instance. FCFS uses the tested native-order/native-admission passthrough; capacity_only isolates W/Gamma admission; buffer uses native max_num_seqs without an extra execution/residency cap.",
        limitations=["Fixed short English prompts and bounded output lengths; not broad production coverage.",
                     "KV is retained on policy pauses; CPU KV offload is not implemented.",
                     "Gamma estimator still has restrictive saturated-window eligibility; report actual buffer/FCFS dwell and stale-capacity fallback rather than assume continuous adaptation.",
                     "Arm order is shuffled within each repetition. Separate service startups showed different greedy outputs even in no-pause warmups, so formal pairs use a single shared model instance."],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()))
    save(root / "progress.json", dict(status="prepared", completed_formal_cases=0, planned_formal_cases=36))


class Suite:
    def __init__(self, root):
        self.root = root
        self.running_container = False
        self.service = None
        self.completed = []
        self.cell = None
        self.launch_index = 0
        self.uid, self.gid = os.getuid(), os.getgid()

    def progress(self, status, **extra):
        save(self.root / "progress.json", dict(status=status, updated_at=time.time(),
            controller_pid=os.getpid(), cell=self.cell, completed_formal_cases=len(self.completed),
            completed_cases=self.completed, planned_formal_cases=36, **extra))

    def wait_resources(self):
        stable = 0
        while stable < 3:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                              "--format=csv,noheader,nounits"], text=True)
            values = [tuple(int(v.strip()) for v in line.split(",")) for line in output.splitlines()]
            idle = sum(index < 6 and memory < 128 and util < 5 for index, memory, util in values) == 6
            stable = stable + 1 if idle else 0
            self.progress("waiting_for_original_six_gpus", gpu_snapshot=values, idle_samples=stable)
            if stable < 3:
                time.sleep(15)
        for port in (18195, 18196):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))

    def start(self, modality, cap, enabled):
        self.wait_resources()
        existing = subprocess.check_output(["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$",
                                           "--format", "{{.Names}}"], text=True).strip()
        if existing:
            raise RuntimeError("owned container name already exists; refusing to replace it")
        subprocess.run(["docker", "run", "-d", "--rm", "--name", CONTAINER, "--network", "host",
            "--shm-size", "8g", "--ulimit", "nofile=65536:65536", "--gpus", '"device=0,1,2,3,4,5"',
            "-v", "/home/zhongyu:/home/zhongyu", "-v", f"{self.root}:{self.root}", "-v", "/data/models:/data/models:ro",
            "-v", f"{self.root / 'framework'}:{self.root / 'framework'}:ro",
            "--entrypoint", "sleep", IMAGE, "infinity"], check=True, capture_output=True)
        self.running_container = True
        self.launch_index += 1
        tag = f"{modality}_{cap}_{'adapter' if enabled else 'native'}_{self.launch_index}"
        trace = self.root / "runtime" / (tag + "_traces")
        cfg = self.root / "runtime" / f"scheduler_{modality}_{cap}.json"
        env = dict(PYTHONPATH=str(self.root / "framework"), STAGE_STUDY_FRAMEWORK=str(self.root / "framework"),
            VLLM_BATCH_INVARIANT="1", VLLM_OMNI_BASELINE_POLICY="native_fcfs",
            VLLM_OMNI_STAGE_STUDY_TRACE=str(json.loads((self.root / 'experiment.json').read_text()).get('detailed_stage_trace', 1)),
            STAGE_QUEUE_TRACE_MODE="stage_study",
            STAGE_QUEUE_TRACE_DIR=str(trace), STAGE_QUEUE_RUN_ID=tag,
            TORCHINDUCTOR_CACHE_DIR=str(self.root / "runtime/inductor"), TRITON_CACHE_DIR=str(self.root / "runtime/triton"))
        if json.loads((self.root / "experiment.json").read_text()).get("benchmark_ordered_submit", False):
            env["VLLM_OMNI_BENCHMARK_ORDERED_SUBMIT"] = "1"
        if enabled:
            env["VLLM_OMNI_CLIENT_BUFFER_CONFIG"] = str(cfg)
        command = ["docker", "exec"]
        for key, value in env.items():
            command += ["-e", f"{key}={value}"]
        command += [CONTAINER, "python", "-u", str(self.root / "framework/experiments/stage_buffer_budget/service_entry.py"),
                    "serve", MODEL, "--omni", "--host", "127.0.0.1", "--port", "18195", "--deploy-config",
                    str(self.root / "runtime" / f"deploy_{modality}_{cap}.yaml")]
        with (self.root / "runtime" / (tag + ".log")).open("w") as log:
            self.service = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        manifest = dict(command=command, max_num_seqs=cap, adapter_enabled=enabled,
            scheduler_config=json.loads(cfg.read_text()) if enabled else {}, trace=str(trace),
            deploy_yaml=(self.root / "runtime" / f"deploy_{modality}_{cap}.yaml").read_text(), started_at=time.time())
        save(self.root / "runtime/active_launch.json", manifest)
        save(self.root / "runtime" / (tag + "_launch.json"), manifest)
        self.progress("starting_model", launch=tag)
        for _ in range(240):
            if self.service.poll() is not None:
                raise RuntimeError(f"service exited: {tag}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18195/health", timeout=2) as response:
                    if response.status == 200:
                        return trace
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError("model readiness")

    def stop(self):
        if self.running_container:
            subprocess.run(["docker", "exec", CONTAINER, "chown", "-R", f"{self.uid}:{self.gid}",
                str(self.root / "runtime"), str(self.root / "runs")], capture_output=True, timeout=60)
            subprocess.run(["docker", "stop", "--timeout", "30", CONTAINER], check=True, capture_output=True, timeout=45)
            self.running_container = False
        if self.service is not None:
            self.service.wait(timeout=10)
            self.service = None

    def run_case(self, name, modality, arm, workload, trace, profile=None, formal=False):
        self.progress("running_case", case=name)
        cmd = [sys.executable, "-m", "tokenflow_client_buffer.run_case", "--root", str(self.root),
               "--name", name, "--modality", modality, "--arm", "fcfs" if arm == "fcfs" else "buffer",
               "--workload", str(workload), "--container", CONTAINER, "--max-inflight", "256", "--timeout", "1800"]
        if profile is not None:
            cmd += ["--capacity-profile", str(profile)]
        if arm == "capacity_only":
            cmd.append("--capacity-only")
        with (self.root / "runtime" / (name + ".runner.log")).open("w") as log:
            subprocess.run(cmd, cwd=self.root / "framework", stdout=log, stderr=subprocess.STDOUT, check=True)
        from .verify_run import verify
        case = self.root / "runs" / name
        evidence = verify(case, trace)
        if evidence["errors"] or evidence["unresolved_pauses"]:
            raise RuntimeError(f"mechanism verification failed: {name}")
        rows = json.loads((case / "client/results.json").read_text())
        if any(r["startup_s"] != r["first_ready_s"] for r in rows):
            raise RuntimeError("unexpected client startup waiting")
        if formal:
            self.completed.append(name)
        return case

    def small_workload(self, modality, cap, kind, count):
        rows = [json.loads(line) for line in (self.root / "workloads" / f"{modality}_0.jsonl").read_text().splitlines()][:count]
        for i, row in enumerate(rows):
            row["arrival_s"] = i * .005
            row["payload"]["omni_scheduling"]["source_request_id"] = f"{modality}-{cap}-{kind}-{i}"
        path = self.root / "workloads" / f"{modality}_{cap}_{kind}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def run(self):
        from .report import aggregate, plots
        manifest = json.loads((self.root / "runtime/source_manifest.json").read_text())
        for name, digest in manifest.items():
            if hashlib.sha256((self.root / "framework" / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("frozen source changed: " + name)
        try:
            for modality in ("text", "audio"):
                for cap in (32, 64):
                    self.cell = f"{modality}_{cap}"
                    # Keep weights/kernels identical across arms. The native-limit
                    # FCFS branch preserves the original native queue and admission.
                    trace = self.start(modality, cap, True)
                    warm = self.small_workload(modality, cap, "warmup", 4)
                    self.run_case(f"{self.cell}_shared_warmup", modality, "fcfs", warm, trace)
                    calibration = self.small_workload(modality, cap, "calibration", cap * 2)
                    case = self.run_case(f"{self.cell}_calibration", modality, "fcfs", calibration, trace)
                    stats = json.loads((case / "client/summary.json").read_text())
                    profile = self.root / "workloads" / f"{self.cell}_capacity.json"
                    first_batch=sorted(json.loads((case/'client/results.json').read_text()),
                                       key=lambda r:r['scheduled_at'])[:cap]
                    save(profile, dict(modality=modality, raw_capacity=stats["generated_units_per_s"],
                        startup_bound_s=max(r['first_ready_s'] for r in first_batch)+.25,
                        units=stats["units"], duration_s=stats["generated_window_s"], max_num_seqs=cap,
                        source_case=str(case), scope="independent native-policy calibration; same six GPUs and bounded workload"))
                    smoke = self.small_workload(modality, cap, "smoke", cap + 8)
                    def hashes(case):
                        return {r['request_id']:r['content_sha256'] for r in json.loads((case/'client/results.json').read_text())}
                    native_smoke = self.run_case(f"{self.cell}_native_smoke", modality, "fcfs", smoke, trace)
                    for arm in ("capacity_only", "buffer"):
                        other = self.run_case(f"{self.cell}_{arm}_smoke", modality, arm, smoke, trace, profile)
                        if hashes(native_smoke) != hashes(other):
                            raise RuntimeError(f"shared-instance native/{arm} smoke output mismatch")
                    for rep in range(3):
                        arms = ["fcfs", "capacity_only", "buffer"]
                        random.Random(92800 + cap + rep).shuffle(arms)
                        paired = {}
                        for arm in arms:
                            paired[arm] = self.run_case(f"{modality}_c{cap}_formal_{rep}_{arm}", modality, arm,
                                self.root / "workloads" / f"{modality}_{rep}.jsonl", trace,
                                None if arm == "fcfs" else profile, formal=True)
                        for arm in ("buffer", "capacity_only"):
                            if hashes(paired['fcfs']) != hashes(paired[arm]):
                                raise RuntimeError(f"formal content mismatch: {modality}/{cap}/{rep}/{arm}")
                    self.stop()
                    prefix = f"{modality}_c{cap}_formal"
                    summary, pairs = aggregate(self.root, prefix, 3)
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
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--previous", type=Path, default=Path("/home/zhongyu/research_reports/tokenflow_client_20261106_4h"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root, args.previous, Path(__file__).resolve().parents[1])
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        Suite(root).run()


if __name__ == "__main__":
    main()
