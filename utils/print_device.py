import os
import torch
# from utils.utils import get_default_device


def get_default_device():
    # [sdaa-adapt] SDAA builds report CUDA as unavailable, so consult the shared
    # accelerator probe before falling back to MPS/CPU.
    from utils.accelerator import get_device as _get_accel_device
    accel_device = _get_accel_device()
    if accel_device.type != 'cpu':
        return accel_device
    if torch.backends.mps.is_available():
        # Not all operations implemented in MPS yet
        use_mps = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        if use_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')


device = get_default_device()
print(f"DiffDock Device: {device}")