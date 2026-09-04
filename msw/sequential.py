"""EXPERIMENTAL: anytime-valid sequential certification.

``SequentialCertifier`` turns the test into a *sequential* one: feed it fresh,
independent batches as they arrive and stop the moment ``certified`` becomes
True -- the false-alarm guarantee holds under optional stopping.

Validity rests on an exact property of the four-term floor-corrected row: under
H0 its distribution is symmetric (swapping one dataset's half-blocks maps the
row t to -t while preserving the joint law), so any odd function f of the row
satisfies E[f] = 0, and for predictable lambda with |lambda * f| <= 1 the
product E_t = prod_i (1 + lambda * f(t_i)) over independent batches is a
nonnegative supermartingale with initial value 1.  Ville's inequality then
bounds P(sup_t E_t >= 1/alpha) <= alpha at ANY data-dependent stopping time.

Requirements: each ``update`` must receive a FRESH pair of batches (never reuse
images), and the per-slice scalings are updated only with past batches (the
first batch is spent on calibration and not bet on).  Pilot numbers (Gaussian
random fields, batch 16/side): anytime false-alarm 0.035-0.040 at alpha=0.05,
100% detection on a slope perturbation the fixed-N test needs ~25 images for,
median stop ~224 images/side with this deliberately simple betting scheme.

Experimental: the betting scheme (uniform slice weights, clipped bets over a
small lambda grid) is the crudest valid choice; power has clear headroom.
"""

from __future__ import annotations

import torch

from .estimators import MultiScaleSW

__all__ = ["SequentialCertifier"]


class SequentialCertifier:
    """Anytime-valid two-sample certification by betting (experimental)."""

    def __init__(self, channels: int, size: int, alpha: float = 0.05,
                 lams: tuple = (0.2, 0.4, 0.6, 0.8), levels: int | None = None,
                 k: int = 5, seed: int = 0):
        if levels is None:
            levels = 4 if size <= 32 else (5 if size <= 64 else 6)
        self.est = MultiScaleSW(channels, size, levels=levels, k=k,
                                n_filters=None, bank="dct", seed=seed)
        self.alpha = alpha
        self.lams = torch.tensor(lams, dtype=torch.float64)
        self.E = torch.ones_like(self.lams)
        self.n_batches = 0
        self.n_images = 0
        self.certified = False
        self._rows = []          # past rows, for the predictable scalings
        self._sig = None         # per-slice scale (predictable)
        self._mhat = None        # typical |sum| scale (predictable)

    @property
    def e_value(self) -> float:
        """Current e-value; >= 1/alpha certifies a difference."""
        return float(self.E.mean())

    def _rescale(self):
        rows = torch.stack(self._rows)
        self._sig = rows.abs().mean(0).clamp_min(1e-30)
        self._mhat = float((rows / self._sig).sum(1).abs().mean()) or 1e-30

    def update(self, a: torch.Tensor, b: torch.Tensor) -> "SequentialCertifier":
        """Consume one FRESH pair of batches; returns self for chaining."""
        dev = self.est.layouts[0].filters.device
        row = self.est.blocked(a.to(dev).float(), b.to(dev).float(),
                               groups=1)[0].double().cpu()
        self.n_batches += 1
        self.n_images += a.shape[0]
        if self._sig is None:                    # calibration batch: no bet
            self._rows.append(row)
            self._rescale()
            return self
        v = float((row / self._sig).sum())
        bet = max(-1.0, min(1.0, v / (3.0 * self._mhat)))
        if not self.certified:
            self.E = self.E * (1 + self.lams * bet)
            if self.e_value >= 1.0 / self.alpha:
                self.certified = True
        self._rows.append(row)                   # update scalings AFTER betting
        self._rescale()
        return self
