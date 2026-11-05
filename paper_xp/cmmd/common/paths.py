"""Where the experiments read and write.

WORK holds the large intermediate data -- COCO-30k at 256 and 512 px, generated
latents, tokens and images, cached features (about 110 GB).  Set MSW_XP_WORK to
a disk with room; it defaults to ./work at the repository root.  Model weights
and the COCO-30k dataset come through the Hugging Face cache (HF_HOME).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("MSW_XP_WORK", os.path.join(REPO, "work"))
DATA, GEN, FEAT = (os.path.join(WORK, d) for d in ("data", "gen", "feats"))
for _d in (DATA, GEN, FEAT):
    os.makedirs(_d, exist_ok=True)


def results(experiment_dir: str) -> str:
    """The results/ folder of an experiment, created on demand."""
    out = os.path.join(experiment_dir, "results")
    os.makedirs(out, exist_ok=True)
    return out
