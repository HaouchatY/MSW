"""results/series_sd_iter.json -> data/sd_iter.csv (+ wrong-way segments per metric)"""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{H}/results/series_sd_iter.json"))
rows = [dict(r, x=x) for x, r in zip(S["x"], S["rows"])]
os.makedirs(f"{H}/data", exist_ok=True)
with open(f"{H}/data/sd_iter.csv", "w") as f:
    f.write("x,fid,cmmd,msw\n")
    for r in rows:
        f.write(f"{r['x']},{r['fid']:.6g},{r['cmmd']:.6g},{r['msw']:.6g}\n")
for m in ("fid", "cmmd", "msw"):          # the images improve with the iteration: values should fall
    with open(f"{H}/data/sd_iter_wrong_{m}.csv", "w") as f:
        f.write("x,y\n")
        for a, b in zip(rows[:-1], rows[1:]):
            if b[m] > a[m]:
                f.write(f"{a['x']},{a[m]:.6g}\n{b['x']},{b[m]:.6g}\n\n")
