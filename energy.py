"""GPU energy per generation, from the GPU's own cumulative energy counter (NVML).

    e0 = gpu_energy_j(); ...generate...; energy_j = gpu_energy_j() - e0

Exact (no sampling) for spans of ~1 s and up. Works for in-process models (HF) and for a
server we call over HTTP (ollama, vLLM): the client waits, the GPU burns the energy.
`uv run energy.py` prints the GPU and one idle second as a sanity check.
"""
import pynvml

pynvml.nvmlInit()
_gpu = pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_energy_j() -> float:
    """Joules the whole GPU has consumed since the driver loaded; subtract two reads for a span.
    shortcut: GPU only (no CPU/RAM) and whole-GPU (display, other processes included) — run on
    an otherwise idle machine; add codecarbon (RAPL) if CPU-side energy ever matters."""
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_gpu) / 1000


if __name__ == "__main__":
    import time

    e0 = gpu_energy_j()
    time.sleep(1)
    idle = gpu_energy_j() - e0
    assert 0 < idle < 100, idle  # a laptop GPU idles at a few watts
    print(f"{pynvml.nvmlDeviceGetName(_gpu)}: idle second = {idle:.1f} J")
