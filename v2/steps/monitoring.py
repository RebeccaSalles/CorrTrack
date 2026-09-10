"""Step 6 — monitoring the persistence of the correlations.

Inspired by v1's 4th phase (`_in_corr` / `_out_corr`) but simplified: tracks,
window after window, the lifetime of every correlated relation and the
transitions (appearance, disappearance, sign change).

A relation is identified by (id1, id2, lag), independently of the absolute time
(lag = |t1 - t2|). This is pure bookkeeping: no heavy computation, hence no
backend dependency.
"""


class Monitor:
    def __init__(self, window_step):
        self.window_step = window_step
        self.active = {}      # key -> {start, last, sign, length}
        self.episodes = []    # closed episodes: (key, start, last, length, sign)
        self.anomalies = []   # (key, time, kind) ; kind in {"in", "out", "sign_flip"}

    def update(self, correlated, current_time):
        """Update the state with the correlated pairs of the current window.

        correlated: list of (keyA, keyB, corr, signed_lag) coming from the
        validation.
        """
        seen = set()
        for a, b, corr, lag in correlated:
            key = (a[0], b[0], lag)  # signed lag -> both directions are distinct
            sign = 1 if corr >= 0 else -1
            seen.add(key)

            ep = self.active.get(key)
            if ep is None:
                self._open(key, current_time, sign)            # apparition
            elif sign != ep["sign"]:
                self._close(key, current_time)                 # changement de signe
                self._open(key, current_time, sign)
                self.anomalies.append((key, current_time, "sign_flip"))
            else:
                ep["last"] = current_time                      # prolongation
                ep["length"] += self.window_step

        # toute relation active non revue ce tour-ci se termine
        for key in [k for k in self.active if k not in seen]:
            self._close(key, current_time)
            self.anomalies.append((key, current_time, "out"))

    def finalize(self):
        """Close the episodes still open at the end of the stream."""
        for key in list(self.active):
            self._close(key, self.active[key]["last"])
        return self.episodes, self.anomalies

    def _open(self, key, time, sign):
        self.active[key] = {"start": time, "last": time, "sign": sign,
                            "length": self.window_step}
        self.anomalies.append((key, time, "in"))

    def _close(self, key, time):
        ep = self.active.pop(key)
        self.episodes.append((key, ep["start"], ep["last"], ep["length"], ep["sign"]))
