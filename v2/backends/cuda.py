"""CUDA backend — STUB (not implemented).

Goal: delegate the primitives to the GPU (cupy / numba.cuda). To be written
later. The class exists and is registered so the swap is immediately possible.
"""

from .base import Backend


class CudaBackend(Backend):
    name = "cuda"

    def project(self, matrix, values):
        raise NotImplementedError("CudaBackend.project — not implemented yet")

    def znormalize(self, vec):
        raise NotImplementedError("CudaBackend.znormalize — not implemented yet")

    def pearson(self, x, y):
        raise NotImplementedError("CudaBackend.pearson — not implemented yet")

    def correlate(self, pairs):
        raise NotImplementedError("CudaBackend.correlate — not implemented yet")

    def cosine(self, u, v):
        raise NotImplementedError("CudaBackend.cosine — not implemented yet")

    def distance(self, u, v):
        raise NotImplementedError("CudaBackend.distance — not implemented yet")
