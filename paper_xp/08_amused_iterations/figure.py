"""results/series_amused_iter.json -> data/amused_iter.csv"""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{H}/results/series_amused_iter.json"))
os.makedirs(f"{H}/data", exist_ok=True)
with open(f"{H}/data/amused_iter.csv", "w") as f:
    f.write("x,fid,cmmd,msw\n")
    for x, r in zip(S["x"], S["rows"]):
        f.write(f"{x},{r['fid']:.6g},{r['cmmd']:.6g},{r['msw']:.6g}\n")
