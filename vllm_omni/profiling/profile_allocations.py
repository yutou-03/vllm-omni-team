"""Metadata observations of the actual allocator checks; no scheduler decisions."""
from __future__ import annotations
import time
from vllm_omni.profiling import stage_study


def install(scheduler):
    manager = getattr(scheduler, 'kv_cache_manager', None)
    if manager is None or getattr(manager, '_profile_installed', False):
        return
    stage = scheduler._omni_stage_id_for_trace()
    state = {'serial': 0, 'current': None, 'last_failure': None}
    original_allocate = manager.allocate_slots
    original_check = manager.coordinator.get_num_blocks_to_allocate
    original_free = manager.free
    original_preempt = scheduler._preempt_request

    def emit(event, request=None, **fields):
        stage_study.emit('profile_' + event, stage_id=stage,
                         request_id=getattr(request, 'request_id', None),
                         observation_basis='native_allocator_metadata', **fields)

    def check(*args, **kwargs):
        count = original_check(*args, **kwargs)
        free = manager.block_pool.get_num_free_blocks()
        current = state['current']
        if current is not None:
            current.update(required_new_blocks=int(count), free_blocks_at_check=int(free),
                           check_monotonic=time.monotonic())
        if current is None and count > free:
            stage_study.emit('profile_capacity_check_failed', stage_id=stage,
                             request_id=kwargs.get('request_id'), required_new_blocks=int(count),
                             free_blocks_at_check=int(free), caller_scope='outside_allocate_slots')
        return count

    def allocate(request, *args, **kwargs):
        state['serial'] += 1
        operation = {'allocation_id': state['serial'], 'trigger_request_id': str(request.request_id),
                     'request_status_before': str(getattr(request.status, 'name', request.status)),
                     'requested_new_tokens': args[0] if args else kwargs.get('num_new_tokens'),
                     'num_computed_tokens_before': int(request.num_computed_tokens),
                     'free_blocks_before': int(manager.block_pool.get_num_free_blocks()),
                     'started_monotonic': time.monotonic()}
        state['current'] = operation
        try:
            result = original_allocate(request, *args, **kwargs)
        except Exception as exc:
            emit('allocation_exception', request, **operation, exception_type=type(exc).__name__)
            raise
        finally:
            state['current'] = None
        operation.update(allocation_succeeded=result is not None,
                         free_blocks_after=int(manager.block_pool.get_num_free_blocks()))
        if result is None:
            state['last_failure'] = dict(operation)
        if result is None or operation.get('required_new_blocks', 0) or operation['free_blocks_after'] != operation['free_blocks_before']:
            emit('allocation_result', request, **operation)
        return result

    def free(request):
        before = stage_study.request_progress(scheduler, request)
        pool_before = manager.block_pool.get_num_free_blocks()
        result = original_free(request)
        emit('kv_free', request, allocated_blocks_before=before.get('allocated_blocks_per_group'),
             free_blocks_before=int(pool_before), free_blocks_after=int(manager.block_pool.get_num_free_blocks()),
             status_before=before.get('status'))
        return result

    def preempt(request, timestamp):
        failure = state['last_failure']
        emit('victim_selected', request, victim_request_id=str(request.request_id),
             triggering_allocation=failure,
             explicit_preemption_cause=getattr(request, '_omni_next_preemption_cause', 'allocation_failure'),
             free_blocks_before=int(manager.block_pool.get_num_free_blocks()))
        return original_preempt(request, timestamp)

    manager.coordinator.get_num_blocks_to_allocate = check
    manager.allocate_slots = allocate
    manager.free = free
    scheduler._preempt_request = preempt
    manager._profile_installed = True
    emit('allocator_observer_installed')
