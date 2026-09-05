# MSW

Multi-scale sliced Wasserstein two-sample tests for stationary image
distributions.

## Install

```bash
pip install git+https://github.com/HaouchatY/MSW.git
```

## Quickstart

```python
import torch, msw

A = torch.rand(64, 3, 32, 32)
B = torch.rand(64, 3, 32, 32)

r = msw.test(A, B)
print(r.T, r.p)                 # statistic, exact p-value
print(r.diagnose())

d, lo, hi = msw.distance(A, B)  # distance with 95% CI
```

More: `examples/quickstart.py`.

MIT license.
