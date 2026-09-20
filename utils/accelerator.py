"""[sdaa-adapt] Accelerator-agnostic device helpers.

Teco SDAA builds of PyTorch expose the accelerator through torch.sdaa while
torch.cuda stays importable but reports is_available() == False. Upstream
DiffDock only probes torch.cuda, so every device decision silently falls back
to CPU on SDAA. Routing the probes through this module leaves the CUDA code
path unchanged and makes SDAA a first-class backend.
"""
import torch

_SDAA = getattr(torch, "sdaa", None)


def _sdaa_ok():
    return _SDAA is not None and _SDAA.is_available()


def accel_available():
    """True when a usable accelerator backend (CUDA or SDAA) is present."""
    return torch.cuda.is_available() or _sdaa_ok()


def device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if _sdaa_ok():
        return _SDAA.device_count()
    return 0


def get_device():
    """torch.device for the active accelerator, CPU when there is none."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _sdaa_ok():
        return torch.device("sdaa")
    return torch.device("cpu")


def empty_cache():
    """Drop cached accelerator memory; no-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif _sdaa_ok():
        _SDAA.empty_cache()


def jit_script_available():
    """[sdaa-adapt] True only when torch.jit.script really compiles.

    Some vendor PyTorch builds ship TorchScript as a no-op: torch.jit.script
    hands back its argument unchanged (and torch.jit.is_jit_enabled may not even
    exist). Probe the behaviour once instead of trusting the module to be there.
    """

    def _probe(x):
        return x + 1

    try:
        scripted = torch.jit.script(_probe)
    except Exception:
        return False
    return isinstance(scripted, (torch.jit.ScriptModule, torch.jit.ScriptFunction))


def apply_platform_defaults():
    """[sdaa-adapt] Apply the library defaults this backend needs.

    e3nn builds its tensor products through FX codegen
    (e3nn/util/codegen/_mixin.py) and asserts that torch.jit.script returned a
    ScriptModule. On a build whose torch.jit is stubbed that assert fires while
    constructing the very first o3.FullyConnectedTensorProduct, so not a single
    score-model layer can be built. e3nn ships an official switch for exactly
    this case: jit_script_fx=False registers the fx.GraphModule directly and
    runs it eagerly, which is numerically identical on both backends.

    Returns True when the workaround was applied, False when nothing was needed.
    """
    if jit_script_available():
        return False
    try:
        import e3nn
    except ImportError:
        return False
    if not hasattr(e3nn, "get_optimization_defaults"):
        return False
    if not e3nn.get_optimization_defaults().get("jit_script_fx", False):
        return False
    e3nn.set_optimization_defaults(jit_script_fx=False)
    return True


# Import-time call: every entry point that touches this module is covered no
# matter where the accelerator import sits in its import list.
apply_platform_defaults()
