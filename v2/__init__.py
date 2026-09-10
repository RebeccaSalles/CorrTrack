"""CorrTrack v2 — modular, simplified rewrite.

Pipeline (see README.md):

    read csv
        chunk csv into sub-windows
            sketch
                -> grid | tree   (interchangeable backends)
            select sketches
                validate sketches   (approximate filter, sketch level)
    and validation                  (exact validation, raw data)

Constraints of this version:
  * standalone (no import from the v1 lib);
  * structures based on pandas / numpy;
  * no parallelization, no vectorization of the engine (explicit loops).
"""
