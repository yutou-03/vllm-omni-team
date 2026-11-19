"""Two-repeat formal comparison plus explicit staggered audio mechanism proof."""
import argparse
import json
from pathlib import Path
import signal
from .focused_suite import FocusedSuite
from .revision_suite import RevisionSuite
from .native_suite import save

class ValidationSuite(RevisionSuite):
    def progress(self,status,**extra):
        FocusedSuite.progress(self,status,**extra)

    def case(self,name,modality,arm,workload,trace,profile=None,formal=False):
        result=FocusedSuite.case(self,name,modality,arm,workload,trace,profile,formal)
        if name=='audio_32_buffer_smoke':
            staggered=self.small_workload('audio',32,'staggered',8)
            rows=[json.loads(line) for line in staggered.read_text().splitlines()]
            for i,row in enumerate(rows):row['arrival_s']=i*.005 if i<4 else 12.+(i-4)*.005
            staggered.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            native=self.case('audio_32_staggered_fcfs','audio','fcfs',staggered,trace)
            buffered=self.case('audio_32_staggered_buffer','audio','buffer',staggered,trace,profile)
            if self.hashes(native)!=self.hashes(buffered):raise RuntimeError('audio diagnostic output mismatch')
            ids=[r['payload']['omni_scheduling']['source_request_id'] for r in rows]
            verified=self.exchange_proof(buffered,trace,set(ids[:4]),set(ids[4:]),1)
            save(self.root/'reports/audio_mechanism.json',dict(verified_exchange=verified,
                identical_outputs=8,case=str(buffered),formal_performance_claim=False))
            if not verified:raise RuntimeError('audio exchange not verified; inspect before formal run')
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    ValidationSuite(a.root.resolve()).run_focused()

if __name__=='__main__':main()
