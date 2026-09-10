"""Apple Neural Engine (ANE) backend through CoreML — EXPERIMENTAL, best-effort.

The ANE is not directly programmable: everything goes through CoreML
(`coremltools`), which alone decides where the ops run (ANE / GPU / CPU) — we
only ALLOW it through `compute_units = CPU_AND_NE`, without being able to force
it. The ANE targets fixed-shape NN inference in float16; our pair batches vary
per window and float16 degrades the correlation precision. This backend is
therefore provided as an experiment.

Safeguards:
  * falls back to the `vectorized` (numpy) backend when `coremltools`/`torch`
    are missing;
  * SELF-VALIDATION on the 1st `correlate`: the CoreML output is compared to
    numpy; if the deviation exceeds the tolerance, CoreML is disabled and numpy
    is kept (avoids silently producing wrong correlations).

    pip install coremltools torch     # to enable the ANE attempt
    python3 -m v2.bf data.csv --backend coreml
"""

import warnings

import numpy as np

from .vectorized import VectorizedBackend

_TOL = 5e-2  # loose tolerance: the ANE computes in float16


class CoremlBackend(VectorizedBackend):
    name = "coreml"

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers, cores)
        self._model = None       # compiled CoreML model (for a window length W)
        self._w = None           # W of the current model
        self._disabled = False   # set to True when unavailable or self-validation failed

    def _build(self, w):
        """Build (torch -> CoreML) a batched Pearson model for W=w."""
        import coremltools as ct
        import torch

        class _Pearson(torch.nn.Module):
            def forward(self, X, Y):
                xc = X - X.mean(dim=1, keepdim=True)
                yc = Y - Y.mean(dim=1, keepdim=True)
                num = (xc * yc).sum(dim=1)
                den = torch.sqrt((xc * xc).sum(dim=1) * (yc * yc).sum(dim=1))
                return num / den

        ex = torch.rand(8, w)
        traced = torch.jit.trace(_Pearson().eval(), (ex, ex))
        batch = ct.RangeDim(1, -1)  # variable number of pairs
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="X", shape=(batch, w)),
                    ct.TensorType(name="Y", shape=(batch, w))],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        return model

    def _ensure(self, w, pairs):
        """Build + self-validate the model for W=w; return True when usable."""
        if self._disabled:
            return False
        if self._model is not None and self._w == w:
            return True
        try:
            self._model = self._build(w)
            self._w = w
            # self-validation against numpy on the current sample
            ref = np.asarray(super().correlate(pairs[:64]))
            got = self._predict([p[0] for p in pairs[:64]], [p[1] for p in pairs[:64]])
            ok = np.allclose(ref, got, atol=_TOL, equal_nan=True)
            if not ok:
                warnings.warn("CoremlBackend: ANE/CoreML output out of tolerance vs numpy "
                              "(float16?) — disabled, falling back to vectorized.",
                              RuntimeWarning, stacklevel=2)
                self._disabled = True
                return False
            return True
        except Exception as exc:
            warnings.warn(f"CoremlBackend : CoreML indisponible ({exc}) — repli sur "
                          "vectorized. `pip install coremltools torch` pour tenter l'ANE.",
                          RuntimeWarning, stacklevel=2)
            self._disabled = True
            return False

    def _predict(self, xs, ys):
        X = np.stack([np.asarray(x, dtype=np.float32) for x in xs])
        Y = np.stack([np.asarray(y, dtype=np.float32) for y in ys])
        out = self._model.predict({"X": X, "Y": Y})
        return np.asarray(next(iter(out.values()))).reshape(-1)

    def correlate(self, pairs):
        if not pairs:
            return []
        w = len(pairs[0][0])
        if not self._ensure(w, pairs):
            return super().correlate(pairs)        # repli numpy
        try:
            return self._predict([p[0] for p in pairs], [p[1] for p in pairs]).tolist()
        except Exception:
            self._disabled = True
            return super().correlate(pairs)
