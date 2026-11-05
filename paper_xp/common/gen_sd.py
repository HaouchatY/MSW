"""Stable Diffusion 1.4 on the COCO-30k captions, 100 DDIM iterations.

Reproduces the generator of CMMD Figs. 6 and 8: the public SD 1.4, no safety
checker and no CLIP re-ranking ("without bells and whistles"), guidance 7.5,
512 px, 100 iterations.  DDIM (eta = 0) makes "iteration k" unambiguous -- the
PNDM default takes one extra warm-up call.

Stored per image: the latents after iterations 95..100 (fp16) for Fig. 6, and
the decoded final image (uint8) for Fig. 8.  The initial noise of image i is
seeded by i alone, so shards are order-independent and a restart resumes
exactly where it stopped.
"""
import argparse, json, os, time

import numpy as np
import torch

from diffusers import DDIMScheduler, StableDiffusionPipeline

from paths import DATA, GEN as OUT
N, STEPS, KEEP = 30000, 100, (95, 96, 97, 98, 99, 100)


def open_stores():
    """The first worker creates the stores (under an atomic lock); the others wait."""
    lat_shape, img_shape = (N, len(KEEP), 4, 64, 64), (N, 512, 512, 3)
    os.makedirs(f"{OUT}/locks", exist_ok=True)
    ready = f"{OUT}/locks/sd_stores/ready"
    try:
        os.mkdir(f"{OUT}/locks/sd_stores")
        np.memmap(f"{OUT}/sd_lat.f16", np.float16, "w+", shape=lat_shape).flush()
        np.memmap(f"{OUT}/sd_final512.u8", np.uint8, "w+", shape=img_shape).flush()
        open(ready, "w").close()
    except FileExistsError:
        while not os.path.exists(ready):
            time.sleep(2)
    return (np.memmap(f"{OUT}/sd_lat.f16", np.float16, "r+", shape=lat_shape),
            np.memmap(f"{OUT}/sd_final512.u8", np.uint8, "r+", shape=img_shape))


SHARD = 1250


def claim():
    """Atomically claim the next unfinished shard (mkdir is atomic)."""
    for s0 in range(0, N, SHARD):
        try:
            os.mkdir(f"{OUT}/locks/sd_{s0}")
            return s0, min(s0 + SHARD, N)
        except FileExistsError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=-1)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--worker", action="store_true", help="claim shards until none is left")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="time this many batches, write nothing")
    a = ap.parse_args()

    caps = json.load(open(f"{DATA}/captions.json"))
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if a.compile:
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet)

    if not a.bench:
        lat_mm, img_mm = open_stores()
    vae = pipe.vae
    t0, n_done, nb = time.time(), 0, 0
    ranges = iter(lambda: claim(), None) if a.worker else iter([(a.start, a.stop)])
    for a.start, a.stop in ranges:
      done_path = f"{OUT}/sd_done_{a.start}_{a.stop}.npy"
      done = np.load(done_path) if os.path.exists(done_path) else np.zeros(a.stop - a.start, bool)
      for b0 in range(a.start, a.stop, a.bs):
          b1 = min(b0 + a.bs, a.stop)
          if done[b0 - a.start:b1 - a.start].all() and not a.bench:
              continue
          idx = list(range(b0, b1))
          gens = [torch.Generator("cuda").manual_seed(i) for i in idx]
          keep = {}

          def cb(p, i, t, kw):
              k = i + 1                                   # iterations completed
              if k in KEEP:
                  keep[k] = kw["latents"].detach().to(torch.float16).cpu()
              return kw

          out = pipe([caps[i] for i in idx], num_inference_steps=STEPS, guidance_scale=7.5,
                     height=512, width=512, generator=gens, output_type="latent",
                     callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
          assert sorted(keep) == list(KEEP), sorted(keep)
          with torch.no_grad():
              x = vae.decode(out.images / vae.config.scaling_factor).sample
          u8 = ((x / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
          nb += 1
          if a.bench:
              torch.cuda.synchronize()
              if nb == 1:
                  t0 = time.time()                        # exclude warm-up / compilation
              elif nb > a.bench:
                  dt = time.time() - t0
                  print(f"bench: {(nb - 1) * a.bs / dt:.3f} img/s (compile={a.compile})", flush=True)
                  return
              continue
          lat_mm[b0:b1] = torch.stack([keep[k] for k in KEEP], 1).numpy()
          img_mm[b0:b1] = u8
          lat_mm.flush(); img_mm.flush()
          done[b0 - a.start:b1 - a.start] = True
          np.save(done_path, done)
          n_done += b1 - b0
          el = time.time() - t0
          print(f"[{a.start}:{a.stop}] {done.sum()}/{len(done)}  {n_done / el:.3f} img/s  "
                f"eta {(len(done) - done.sum()) / max(n_done / el, 1e-9) / 3600:.2f} h", flush=True)
      print("SD SHARD DONE", a.start, a.stop, flush=True)


if __name__ == "__main__":
    main()
