"""
CUDA Stress Test for Defocus Model

This script tests if the model can run stable forward/backward passes on GPU.
Used to debug intermittent "illegal memory access" CUDA errors.

Run with: python test_cuda.py
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force synchronous CUDA ops for debugging

import torch
import time
from model import DefocusNet

# =========================================================================
# CUDA STABILITY SETTINGS
# All of these were tried to fix "illegal memory access" errors:
# - benchmark=False: Disables cuDNN autotuning
# - deterministic=True: Forces deterministic algorithms
# Result: Errors still occur at random batches during training
# =========================================================================
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model = DefocusNet().cuda()
optimizer = torch.optim.Adam(model.dme_subnet.parameters())

print("Running CUDA stress test on DME-subnet...")
print("If this passes but training fails, the issue is likely in:")
print("  - DataLoader interactions")
print("  - Memory fragmentation during long runs")
print("  - Specific tensor operations in training loop")
print()

for bs in [4, 8, 16, 32, 64]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(bs, 1, 296, 296, device="cuda")

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            out = model.dme_subnet(x)
            loss = out.mean()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        reserved = torch.cuda.max_memory_reserved() / 1e9
        allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"Batch {bs}: {50/elapsed:.1f} it/s, alloc {allocated:.2f}GB, reserved {reserved:.2f}GB")

    except Exception as e:
        print(f"\nFAILED at batch {bs}: {type(e).__name__}: {e}\n")
        raise

print("\nCUDA stress test PASSED")
print("GPU is working correctly for isolated forward/backward passes")
