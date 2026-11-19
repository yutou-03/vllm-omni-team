"""Benchmark-only nonblocking server queue order; no GPU acknowledgement fence."""
import threading

class SubmissionOrder:
    def __init__(self):
        self.groups = {}
        self.lock = threading.Lock()

    def submit(self, group, rank, message, send):
        if not isinstance(group, str) or not group or type(rank) is not int or rank < 0:
            raise ValueError('invalid benchmark sequence')
        with self.lock:
            state = self.groups.setdefault(group, dict(next=0, pending={}, cancelled=False))
            if state['cancelled']:
                raise RuntimeError('benchmark sequence cancelled; use a new session')
            if rank < state['next'] or rank in state['pending']:
                raise ValueError('duplicate benchmark sequence rank')
            if len(state['pending']) >= 10000:
                raise RuntimeError('benchmark reorder buffer exceeded')
            state['pending'][rank] = message
            while state['next'] in state['pending']:
                send(state['pending'][state['next']])
                del state['pending'][state['next']]
                state['next'] += 1

    def abort(self, request_ids):
        ids = set(request_ids)
        with self.lock:
            for state in self.groups.values():
                if any(m['request_id'] in ids for m in state['pending'].values()):
                    state['pending'].clear()
                    state['cancelled'] = True

def submit_benchmark_message(engine, message):
    import os
    if os.environ.get('VLLM_OMNI_BENCHMARK_ORDERED_SUBMIT') != '1':
        return False
    original = message.get('original_prompt')
    if not isinstance(original, dict):
        return False
    meta = (original.get('additional_information') or {}).get('meta', {})
    group = meta.get('sched_service_class', '')
    if not group.startswith('tokenflow_benchmark:'):
        return False
    if engine.prompt_expand_func is not None:
        raise ValueError('ordered benchmark does not support CFG companions')
    if not hasattr(engine, '_benchmark_submission_order'):
        engine._benchmark_submission_order = SubmissionOrder()
    engine._benchmark_submission_order.submit(group, meta.get('sched_ingress_order'),
                                             message, engine.request_queue.sync_q.put_nowait)
    return True
