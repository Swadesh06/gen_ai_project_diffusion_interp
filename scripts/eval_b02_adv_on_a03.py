"""Eval B02-adv detector on held-out A03 embedding-PGD bypass."""
from pathlib import Path
import json, sys, torch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def load_attack_features(d):
    base = Path(d)
    feats = []
    for atk in sorted(base.glob("*.attack.json")):
        try:
            r = json.loads(atk.read_text())
        except Exception:
            continue
        if not r.get("pre_flagged") or r.get("post_flagged"):
            continue
        seed = r.get("seed")
        sae_path = base / "sae" / f"{seed:08d}.sae.pt"
        if not sae_path.exists():
            continue
        try:
            d = torch.load(sae_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        f = []
        for hp in HOOKPOINTS:
            if hp not in d: f = None; break
            v = d[hp]
            if v.ndim == 3:
                v = v.mean(dim=(0, 1))
            elif v.ndim == 4:
                v = v.mean(dim=(1, 2))[0]
            f.append(v.float())
        if f is None: continue
        feats.append(torch.cat(f))
    return feats


# Load B02-adv
ck = torch.load("checkpoints/B02_adv_v1/best.pt", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
W = sd["weight"].numpy()
b = sd["bias"].numpy()
print(f"  B02-adv weight shape: {W.shape}")

# Load A03 5-seed bypass features
all_feats = []
for s in range(5):
    d = f"outputs/A03_emb_eps0.5_n200" if s == 0 else f"outputs/A03_emb_eps0.5_n200_seed{s}"
    if not Path(d).exists():
        print(f"  {d}: missing"); continue
    feats = load_attack_features(d)
    print(f"  {d}: {len(feats)} bypass")
    all_feats.extend(feats)

if not all_feats:
    print("no A03 features found"); sys.exit(1)
X_a03 = torch.stack(all_feats).numpy()
print(f"  X_a03 total shape: {X_a03.shape}")

logits = X_a03 @ W.T + b
n_flag = int((logits > 0).sum())
print(f"==== B02-adv on A03 5-seed bypass ====")
print(f"  n A03 bypass: {len(all_feats)}")
print(f"  n flagged @ logit>0: {n_flag} = {n_flag/len(all_feats):.3f}")
print(f"  mean logit: {logits.mean():.3f}, std: {logits.std():.3f}")

# Compare to B02-v3 (in-distribution baseline) on same features
ck_v3 = torch.load("/workspace/checkpoints/B02_oracle_v3_linear_cat/best.pt", map_location="cpu", weights_only=False)
sd_v3 = ck_v3["model_state_dict"]
W_v3 = sd_v3["weight"].numpy()
b_v3 = sd_v3["bias"].numpy()

logits_v3 = X_a03 @ W_v3.T + b_v3
n_flag_v3 = int((logits_v3 > 0).sum())
print(f"==== B02-v3 (oracle) on same A03 5-seed ====")
print(f"  n flagged @ logit>0: {n_flag_v3} = {n_flag_v3/len(all_feats):.3f}")
print(f"  mean logit: {logits_v3.mean():.3f}, std: {logits_v3.std():.3f}")
