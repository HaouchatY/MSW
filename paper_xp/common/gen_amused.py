"""aMUSEd-256 on the COCO-30k captions: the public stand-in for Muse.

Muse itself is internal to Google.  aMUSEd (Patil et al. 2024) is the open
reproduction of the same design -- a masked-token transformer refining a
VQGAN token grid in parallel over a fixed number of iterations, with an
8192-entry codebook like Muse's -- so it supports the two Muse experiments that
need a generator: the per-iteration curve (CMMD Figs. 3-4) and full against
early-stopped sampling (Table 3, metric columns).

Default sampler: 12 iterations, guidance 10.  At every iteration we keep the
scheduler's `pred_original_sample` -- the committed tokens plus the current
prediction for every still-masked token, i.e. the image the model would emit
if stopped there.  At the last iteration it equals the final sample.
Tokens are stored (int16, 16 x 16 per image per iteration); images are decoded
later, in the measurement pass.
"""
import json, os, time

import numpy as np
import torch

from diffusers import AmusedPipeline

from paths import DATA, GEN as OUT
N, STEPS, BS = 30000, 12, 64


def main():
    caps = json.load(open(f"{DATA}/captions.json"))
    pipe = AmusedPipeline.from_pretrained("amused/amused-256", variant="fp16",
                                          torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    path = f"{OUT}/amused_tok.i16"
    tok = np.memmap(path, np.int16, "r+" if os.path.exists(path) else "w+", shape=(N, STEPS, 16, 16))
    done_path = f"{OUT}/amused_done.npy"
    done = np.load(done_path) if os.path.exists(done_path) else np.zeros(N, bool)

    rec = []
    step = pipe.scheduler.step

    def recording_step(*args, **kw):
        out = step(*args, **kw)
        rec.append(out.pred_original_sample.detach().clone())
        return out

    pipe.scheduler.step = recording_step
    t0, n_done = time.time(), 0
    for b0 in range(0, N, BS):
        b1 = min(b0 + BS, N)
        if done[b0:b1].all():
            continue
        rec.clear()
        g = torch.Generator("cuda").manual_seed(1_000_000 + b0)
        final = pipe([caps[i] for i in range(b0, b1)], num_inference_steps=STEPS,
                     guidance_scale=10.0, generator=g, output_type="latent").images
        assert len(rec) == STEPS, len(rec)
        assert torch.equal(rec[-1].view_as(final), final), "last preview must be the sample"
        assert int(rec[-1].max()) < 8192, "a mask token survived the last iteration"
        tok[b0:b1] = torch.stack([r.view(b1 - b0, 16, 16) for r in rec], 1).cpu().numpy().astype(np.int16)
        tok.flush()
        done[b0:b1] = True
        np.save(done_path, done)
        n_done += b1 - b0
        print(f"{done.sum()}/{N}  {n_done / (time.time() - t0):.1f} img/s", flush=True)
    print("AMUSED DONE", flush=True)


if __name__ == "__main__":
    main()
