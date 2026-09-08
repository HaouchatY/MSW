"""Run the suite on the CPU by default: it is small, and CPU is the floor the
library must work on (no test here needs a GPU).  Set MSW_TEST_DEVICE=cuda to
run the same tests on the GPU instead."""
import os

import msw

msw.set_device(os.environ.get("MSW_TEST_DEVICE", "cpu"))
