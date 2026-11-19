"""Read terminal backend phase events; never supply client buffer credit."""
import json
import math
from pathlib import Path


class PhaseAccounting:
    def init_phase_accounting(self):
        self.phase_offset = 0
        self.phase_inode = None
        self.phase_attempts = {}

    def process_phase_events(self, now):
        text = self.gateways['text']
        path = Path(str(text.config['execution_telemetry_path']) + '.completed.jsonl')
        try:
            with path.open('rb') as stream:
                import os
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if self.phase_inode != identity or stat.st_size < self.phase_offset:
                    self.phase_inode, self.phase_offset = identity, 0
                stream.seek(self.phase_offset)
                # Bound work per loop, preserve partial append for the next read.
                for _ in range(4096):
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        self.phase_offset = start
                        break
                    self.phase_offset = stream.tell()
                    try:
                        event = json.loads(line)
                        route, rid = event['route'], event['request_id']
                        gateway = self.gateways[route]
                        registered = gateway.registered_at.get(rid)
                        began = float(event['started_at'])
                        if (event['stage'] != 0 or event['session'] != text.session
                                or registered is None or not math.isfinite(began)
                                or not registered <= began <= now
                                or gateway.w.requests[rid].state == 'finished'):
                            continue
                        attempt = (began, event['backend_request_id'])
                        old = self.phase_attempts.get(rid)
                        if event['event'] == 'started':
                            if old is None or attempt[0] > old[0]:
                                self.phase_attempts[rid] = attempt
                                self.book.thinker_completed.discard(rid)
                                self.event('resource_phase_started', request_id=rid, attempt=attempt)
                        elif (event['event'] == 'completed' and old == attempt
                                and event['finish_reason'] in ('stop', 'length')
                                and began <= float(event['completed_at']) <= now):
                            before = self.book.demand
                            if self.book.complete_thinker(rid):
                                self.event('resource_phase_completed', request_id=rid,
                                    attempt=attempt, demand_before=before, demand_after=self.book.demand)
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            return
