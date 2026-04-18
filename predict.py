import argparse
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Predict FL architecture from captured features")
parser.add_argument("--features",  default="features.csv", help="Feature CSV from extract_features.py")
parser.add_argument("--model-dir", default="model",        help="Directory containing model.pkl")
parser.add_argument("--archs",     default=None,           help="Comma-separated list of architectures to evaluate (default: all)")
args = parser.parse_args()

print(f"[predict] Model:    {args.model_dir}/model.pkl")
print(f"[predict] Features: {args.features}")

bundle = joblib.load(f"{args.model_dir}/model.pkl")
df     = pd.read_csv(args.features)
print(f"[predict] Loaded {len(df)} windows across {df['arch'].nunique()} architectures")

if len(df) == 0:
    print("[predict] ERROR: No windows found. Check window/min-packets settings match training.")
    exit(1)

if args.archs:
    selected = [a.strip() for a in args.archs.split(',')]
    df = df[df['arch'].isin(selected)].copy()
    print(f"[predict] Filtering to: {selected}")
    print(f"[predict] Remaining windows: {len(df)}")
    if len(df) == 0:
        print("[predict] ERROR: No windows found for the selected architectures.")
        exit(1)

ARCH_NAMES   = bundle["arch_names"]
flow_cols    = bundle["flow_cols"]
pkt_cols     = bundle["pkt_cols"]
best_variant = bundle["best_variant"]

# Verify feature columns match training
missing = [c for c in flow_cols + pkt_cols if c not in df.columns]
if missing:
    print(f"[predict] ERROR: Feature mismatch — {len(missing)} missing columns.")
    print("[predict] Hint: window size and min-packets must match the training config.")
    exit(1)

X_flow = df[flow_cols].values.astype(np.float32)
X_pkt  = df[pkt_cols].values.astype(np.float32)

Xf = bundle["scaler_flow"].transform(X_flow)
Xp = bundle["scaler_pkt"].transform(X_pkt)

# Predict per window using the best variant from training
print(f"[predict] Using variant: {best_variant}")
if best_variant == "flow_only":
    probas = bundle["rf_flow"].predict_proba(Xf)
elif best_variant == "packet_only":
    probas = bundle["rf_pkt"].predict_proba(Xp)
elif best_variant == "fusion_MetaLR":
    meta   = np.hstack([bundle["rf_flow"].predict_proba(Xf), bundle["rf_pkt"].predict_proba(Xp)])
    probas = bundle["meta_lr"].predict_proba(meta)
else:  # fusion_MetaXGB
    meta   = np.hstack([bundle["rf_flow"].predict_proba(Xf), bundle["rf_pkt"].predict_proba(Xp)])
    probas = bundle["meta_xgb"].predict_proba(meta)

preds = np.argmax(probas, axis=1)

# Group predictions by true architecture (known from pcap filename label)
arch_data = {}
for i, (_, row) in enumerate(df.iterrows()):
    arch       = str(row.get("arch", "unknown"))
    pred_label = int(preds[i])
    pred_name  = ARCH_NAMES.get(pred_label, f"class_{pred_label}")

    if arch not in arch_data:
        arch_data[arch] = {
            "true_arch":  arch,
            "n_windows":  0,
            "votes":      {},
            "proba_sums": np.zeros(len(ARCH_NAMES)),
        }
    r = arch_data[arch]
    r["n_windows"]            += 1
    r["votes"][pred_name]      = r["votes"].get(pred_name, 0) + 1
    r["proba_sums"]           += probas[i]

# Summarize per architecture
print(f"\n{'Arch':>12}   {'Predicted':>12}   {'Windows':>10}   Result")
print("-" * 55)

per_arch  = {}
n_correct = 0
for arch, r in sorted(arch_data.items()):
    top_pred  = max(r["votes"], key=r["votes"].get)
    n         = r["n_windows"]
    avg_p     = (r["proba_sums"] / n).tolist()
    correct   = (top_pred.lower() == arch.lower())
    if correct:
        n_correct += 1

    per_arch[arch] = {
        "true_arch":  arch,
        "predicted":  top_pred,
        "correct":    correct,
        "confidence": round(r["votes"][top_pred] / n, 3),
        "n_windows":  n,
        "votes":      r["votes"],
        "avg_probas": {ARCH_NAMES[k]: round(avg_p[k], 4) for k in range(len(ARCH_NAMES))},
    }
    verdict = "✓" if correct else "✗"
    print(f"  {arch:>10}   {top_pred:>12}   "
          f"{r['votes'][top_pred]}/{n} ({r['votes'][top_pred]/n*100:.0f}%)   {verdict}")

total    = len(per_arch)
accuracy = n_correct / total if total > 0 else 0
print(f"\n[predict] Overall: {n_correct}/{total} correct ({accuracy*100:.0f}%)")

output = {
    "model_info": {
        "best_variant":    best_variant,
        "trained_at":      bundle.get("trained_at", "unknown"),
        "cv_f1_mean":      bundle.get("cv_f1_mean", 0),
        "cv_f1_std":       bundle.get("cv_f1_std", 0),
        "n_train_windows": bundle.get("n_windows", 0),
    },
    "per_arch": per_arch,
    "overall":  {
        "correct":  n_correct,
        "total":    total,
        "accuracy": round(accuracy, 3),
    },
}

with open("prediction_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved: prediction_results.json")
