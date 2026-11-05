"""results/table2.json -> data/figure.csv"""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{H}/data", exist_ok=True)
with open(f"{H}/data/figure.csv", "w") as f:
    f.write("lam,fd,fd_inf,mmd,mmd_sd,mmd_pop,msw,msw_sd,msw_pop\n")
    for r in json.load(open(f"{H}/results/table2.json")):
        f.write(f"{r['lam']:.4f},{r['fd']:.6f},{r['fd_inf']:.6f},{r['mmd']:.5f},{r['mmd_sd']:.5f},"
                f"{r['pop_mmd']:.5f},{r['msw']:.6f},{r['msw_sd']:.6f},{r['pop_msw']:.6f}\n")
