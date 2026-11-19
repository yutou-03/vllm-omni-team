"""Capacity-fix validation: staged audio mechanism check, then two text pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

from tokenflow_client_buffer.focused_suite import FocusedSuite
from tokenflow_client_buffer.native_suite import save


class RevisionSuite(FocusedSuite):
    def progress(self, status, **extra):
        save(self.root / "progress.json", dict(status=status, updated_at=time.time(),
            controller_pid=os.getpid(), cell=self.cell, completed_formal_cases=len(self.completed),
            completed_cases=self.completed, planned_formal_cases=4, **extra))

    @staticmethod
    def hashes(case):
        return {r["request_id"]: r["content_sha256"] for r in
                json.loads((case / "client/results.json").read_text())}

    def exchange_proof(self, case, trace, early, later, stage):
        """Require an actual later request to run while an earlier one is held."""
        from tokenflow_client_buffer.verify_run import iter_read
        events = list(iter_read(trace / "stage_events.jsonl"))
        identities = {e['request_id']: e['source_request_id'] for e in events
                      if e['event'] == 'client_buffer_contract'}
        pending, intervals = {}, []
        for e in events:
            rid = e.get('request_id')
            if e['event'] == 'client_buffer_pause' and identities.get(rid) in early:
                pending[rid] = e
            elif e['event'] == 'client_buffer_fallback_release':
                pending.pop(rid, None)
            elif e['event'] == 'client_buffer_resumed' and rid in pending:
                intervals.append((pending.pop(rid), e))
        proof = []
        for step in iter_read(trace / 'iteration_events.jsonl'):
            if step['stage_id'] != stage:
                continue
            successors = [rid for rid in step['scheduled_req_ids'] if identities.get(rid) in later]
            if not successors:
                continue
            at = step.get('schedule_monotonic_end_s', step['monotonic'])
            for pause, resume in intervals:
                if (pause['monotonic'] <= at < resume['monotonic']
                        and pause['request_id'] not in step['scheduled_req_ids']):
                    proof.append(dict(paused=identities[pause['request_id']],
                        successors=[identities[rid] for rid in successors],
                        coverage_s=pause['coverage_s'], threshold_s=pause.get('safety_threshold_s'),
                        execution_limit=pause.get('execution_limit'),
                        selected_before_pause=pause.get('selected_before_pause'),
                        scheduled_step=step['iteration_id'], held_s=resume['held_s']))
        save(case / 'early_late_exchange.json', dict(exchanges=proof, verified=bool(proof)))
        return bool(proof)

    def run_revision(self):
        from tokenflow_client_buffer.report import aggregate, plots
        manifest = json.loads((self.root / 'runtime/source_manifest.json').read_text())
        for name, digest in manifest.items():
            if hashlib.sha256((self.root / 'framework' / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('frozen source changed: ' + name)
        try:
            self.cell = 'audio_32_mechanism'
            trace = self.start('audio', 32, True)
            warm = self.small_workload('audio', 32, 'warmup', 4)
            self.case('audio_32_shared_warmup', 'audio', 'fcfs', warm, trace)
            workload = self.small_workload('audio', 32, 'staggered', 8)
            rows = [json.loads(line) for line in workload.read_text().splitlines()]
            for i, row in enumerate(rows):
                row['arrival_s'] = i * .005 if i < 4 else 12. + (i-4)*.005
            workload.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            profile = self.root / 'workloads/audio_32_capacity.json'
            native = self.case('audio_32_staggered_fcfs', 'audio', 'fcfs', workload, trace)
            buffered = self.case('audio_32_staggered_buffer', 'audio', 'buffer', workload, trace, profile)
            if self.hashes(native) != self.hashes(buffered):
                raise RuntimeError('staggered audio output mismatch')
            ids = [r['payload']['omni_scheduling']['source_request_id'] for r in rows]
            verified = self.exchange_proof(buffered, trace, set(ids[:4]), set(ids[4:]), 1)
            save(self.root / 'reports/audio_mechanism.json', dict(verified_exchange=verified,
                identical_outputs=8, case=str(buffered), formal_performance_claim=False))
            self.stop()

            self.cell = 'text_32'
            trace = self.start('text', 32, True)
            warm = self.small_workload('text', 32, 'warmup', 4)
            self.case('text_32_shared_warmup', 'text', 'fcfs', warm, trace)
            calibration = self.small_workload('text', 32, 'calibration', 64)
            case = self.case('text_32_calibration', 'text', 'fcfs', calibration, trace)
            stats = json.loads((case / 'client/summary.json').read_text())
            profile = self.root / 'workloads/text_32_capacity.json'
            save(profile, dict(modality='text',raw_capacity=stats['generated_units_per_s'],
                units=stats['units'],duration_s=stats['generated_window_s'],max_num_seqs=32,
                source_case=str(case),scope='independent native-policy calibration'))
            smoke = self.small_workload('text', 32, 'smoke', 40)
            native = self.case('text_32_native_smoke', 'text', 'fcfs', smoke, trace)
            buffered = self.case('text_32_buffer_smoke', 'text', 'buffer', smoke, trace, profile)
            if self.hashes(native) != self.hashes(buffered):
                raise RuntimeError('text smoke output mismatch')
            mechanism = json.loads((buffered / 'mechanism.json').read_text())
            if not mechanism['evidenced_substitutions']:
                raise RuntimeError('text mechanism absent; do not run formal comparisons')
            for rep in range(2):
                arms = ['fcfs', 'buffer']
                random.Random(92832+rep).shuffle(arms)
                paired = {}
                for arm in arms:
                    paired[arm] = self.case(f'text_c32_formal_{rep}_{arm}', 'text', arm,
                        self.root / 'workloads' / f'text_{rep}.jsonl', trace,
                        None if arm == 'fcfs' else profile, formal=True)
                if self.hashes(paired['fcfs']) != self.hashes(paired['buffer']):
                    raise RuntimeError('text formal output mismatch')
            self.stop()
            summary, pairs = aggregate(self.root, 'text_c32_formal', 2)
            plots(self.root, 'text_c32_formal', summary, pairs)
            self.progress('complete', audio_mechanism_verified=verified)
        except BaseException as error:
            self.progress('stopped_with_error', error=repr(error))
            raise
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    RevisionSuite(args.root.resolve()).run_revision()


if __name__ == '__main__':
    main()
