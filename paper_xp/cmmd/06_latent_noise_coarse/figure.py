"""results/series_{coco,amused}_vq.json -> data/{coco,amused}.csv"""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{H}/data", exist_ok=True)
for key, name in (("coco_vq", "coco"), ("amused_vq", "amused")):
    S = json.load(open(f"{H}/results/series_{key}.json"))
    with open(f"{H}/data/{name}.csv", "w") as f:
        f.write("p,fid,cmmd,msw\n")
        for x, r in zip(S["x"], S["rows"]):
            f.write(f"{x},{r['fid']:.6g},{r['cmmd']:.6g},{r['msw']:.6g}\n")
