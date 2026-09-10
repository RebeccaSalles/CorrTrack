"""FilCorr — Python port of the UNM algorithm (ICDM 2020).

**Self-registration**: importing this module installs the hooks needed for
`python -m v2.pipeline` to recognize `"mode": "filcorr"` in the JSON. No
external wrapper to use.

The hooks installed (idempotent):

1. `core.pipeline.run` is patched to dispatch `mode='filcorr'` to
   `v2.fillcorr.runner.run`; the other modes are left untouched.
2. `v2.pipeline._norm_params` (when already imported) lets the `filcorr_*` keys
   through without a warning.
3. `core.config.Config.build` absorbs the `filcorr_*` keys as `setattr` on the
   Config (outside the dataclass) → the runner reads them through `getattr`.

See `v2/fillcorr.md` (at the v2/ root) for the user documentation.
"""

from .algo import FilCorrParams, band_indices
from .runner import run

__all__ = ["FilCorrParams", "band_indices", "run"]


def _install_hooks():
    """Install the required patches (idempotent)."""
    from ..core import pipeline as _core_pipeline

    # --- 1. dispatch mode='filcorr' from core.pipeline.run ---
    if not getattr(_core_pipeline.run, "__filcorr_patched__", False):
        _orig_run = _core_pipeline.run

        def _patched_run(config, path, mode="corrtrack", step="all"):
            if mode == "filcorr":
                return run(config, path, mode=mode, step=step)
            return _orig_run(config, path, mode=mode, step=step)

        _patched_run.__filcorr_patched__ = True
        _core_pipeline.run = _patched_run
        # `v2.pipeline` aliases `run_pipeline = core.pipeline.run` AT ITS OWN
        # IMPORT (before us) → that alias still points at the original. Rebind it
        # in every candidate module. Under `python -m v2.pipeline` the module is
        # registered as "__main__" (not "v2.pipeline").
        try:
            import sys
            for modname in ("v2.pipeline", "__main__"):
                mod = sys.modules.get(modname)
                if mod is None:
                    continue
                if hasattr(mod, "run_pipeline"):
                    mod.run_pipeline = _patched_run
        except Exception:
            pass

    # --- 2. Config.build absorbs the filcorr_* (setattr on the cfg) ---
    from ..core.config import Config as _Config
    # NB: `Config.build` is a classmethod. Accessing it through the class yields
    # a BOUND function, whose attributes are not the underlying func; the marker
    # is therefore stored on _Config itself for idempotence.
    if not getattr(_Config, "__filcorr_build_patched__", False):
        _orig_build = _Config.__dict__["build"].__func__   # la vraie fonction

        @classmethod
        def patched_build(cls, **overrides):
            fil = {k: overrides.pop(k) for k in list(overrides)
                   if k.startswith("filcorr_")}
            cfg = _orig_build(cls, **overrides)
            for k, v in fil.items():
                setattr(cfg, k, v)
            return cfg

        _Config.build = patched_build
        _Config.__filcorr_build_patched__ = True

    # --- 3. v2.pipeline._norm_params lets the filcorr_* through without a WARNING ---
    # Subtlety: `python -m v2.pipeline` loads the module as `__main__`, not as
    # "v2.pipeline" → both are checked. Every module exposing a `_norm_params`
    # coming from the v2 package is scanned too (in case the import comes from
    # another entry point).
    try:
        import sys

        def _build_patched(_orig_norm):
            def patched_norm(params):
                fil = {k: v for k, v in params.items() if k.startswith("filcorr_")}
                regular = {k: v for k, v in params.items() if not k.startswith("filcorr_")}
                out = _orig_norm(regular)
                out.update(fil)
                return out
            patched_norm.__filcorr_patched__ = True
            return patched_norm

        for modname in ("v2.pipeline", "__main__"):
            mod = sys.modules.get(modname)
            if mod is None:
                continue
            norm = getattr(mod, "_norm_params", None)
            if norm is None or getattr(norm, "__filcorr_patched__", False):
                continue
            mod._norm_params = _build_patched(norm)
    except Exception:
        pass


_install_hooks()
