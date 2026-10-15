"""Bound CPU tensor backing storage before worker results cross process boundaries."""
import torch


def compact_cpu_tensor_storage(payload):
    """Preserve payload values while removing unrelated storage from CPU views.

    A contiguous slice can still own a view of the full batch's storage.
    PyTorch's pickle reduction serializes that entire storage for each view;
    ``contiguous()`` alone does not fix it. Clone only CPU strided tensors
    whose backing storage exceeds their logical data or whose layout is not
    compact. Do not copy GPU tensors or alter dtype/shape/non-tensor values.
    Repeated references to the same tensor within a payload stay shared.
    """
    memo = {}

    def compact(value):
        if isinstance(value, torch.Tensor):
            identity = id(value)
            if identity not in memo:
                result = value
                if value.device.type == "cpu" and value.layout == torch.strided:
                    logical_bytes = value.numel() * value.element_size()
                    if (value.storage_offset() != 0 or not value.is_contiguous()
                            or value.untyped_storage().nbytes() != logical_bytes):
                        result = value.clone(memory_format=torch.contiguous_format)
                memo[identity] = result
            return memo[identity]
        if isinstance(value, dict):
            return {key: compact(child) for key, child in value.items()}
        if isinstance(value, list):
            return [compact(child) for child in value]
        if isinstance(value, tuple):
            return tuple(compact(child) for child in value)
        return value

    return compact(payload)
