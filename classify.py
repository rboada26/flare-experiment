import argparse
import json
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Classify FL traffic fingerprints")
parser.add_argument("--n-estimators", type=int, default=200,
                    help="Number of trees in Random Forest (default: 200)")
args = parser.parse_args()
N_ESTIMATORS = args.n_estimators

print(f"[classify] n_estimators={N_ESTIMATORS}")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")

ARCH_NAMES = {0:"SimpleCNN", 1:"ResNet18", 2:"MobileNet",
              3:"GRU", 4:"LSTM", 5:"BiLSTM"}

FLOW_COLS = [c for c in df.columns if c.startswith(
    ("pkt_rate","up_down","up_","down_","iat_"))]
PKT_COLS  = [c for c in df.columns if c.startswith(
    ("hist_","first_","last_"))]

X_flow = np.nan_to_num(df[FLOW_COLS].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_pkt  = np.nan_to_num(df[PKT_COLS].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
y      = df["label"].values

print(f"Dataset: {len(df)} windows | {len(FLOW_COLS)} flow features | {len(PKT_COLS)} packet features")
print(f"Classes: {np.bincount(y)}")
print(f"Labels:  {list(ARCH_NAMES.values())}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def multiclass_scores(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred,        average="macro", zero_division=0),
    }

# ── Cross-validation ──────────────────────────────────────────────────────────
n_splits = min(5, int(np.unique(y, return_counts=True)[1].min()))
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

variants = ["flow_only", "packet_only", "fusion_MetaLR", "fusion_MetaXGB"]
fold_records = []
all_y_true, all_y_pred = {v: [] for v in variants}, {v: [] for v in variants}

def _scale(train, test=None):
    """Fit scaler on train, transform both; replace any residual NaN/inf."""
    sc = StandardScaler()
    tr = np.nan_to_num(sc.fit_transform(train))
    if test is None:
        return tr
    return tr, np.nan_to_num(sc.transform(test))

for fold, (train_idx, test_idx) in enumerate(cv.split(X_flow, y)):
    Xf_tr, Xf_te = _scale(X_flow[train_idx], X_flow[test_idx])
    Xp_tr, Xp_te = _scale(X_pkt[train_idx],  X_pkt[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    # Base Random Forests
    rf_flow = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_pkt  = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_flow.fit(Xf_tr, y_tr)
    rf_pkt.fit(Xp_tr, y_tr)

    # Meta features
    meta_tr = np.hstack([rf_flow.predict_proba(Xf_tr), rf_pkt.predict_proba(Xp_tr)])
    meta_te = np.hstack([rf_flow.predict_proba(Xf_te), rf_pkt.predict_proba(Xp_te)])

    # Meta classifiers
    meta_lr  = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000, C=1.0)
    meta_xgb = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS, random_state=42, max_depth=3)
    meta_lr.fit(meta_tr, y_tr)
    meta_xgb.fit(meta_tr, y_tr)

    preds = {
        "flow_only":    rf_flow.predict(Xf_te),
        "packet_only":  rf_pkt.predict(Xp_te),
        "fusion_MetaLR":  meta_lr.predict(meta_te),
        "fusion_MetaXGB": meta_xgb.predict(meta_te),
    }

    for v, y_pred in preds.items():
        s = multiclass_scores(y_te, y_pred)
        fold_records.append({"fold": fold+1, "variant": v, **s})
        all_y_true[v].extend(y_te)
        all_y_pred[v].extend(y_pred)

# ── Summary table ─────────────────────────────────────────────────────────────
detail_df = pd.DataFrame(fold_records)

print("=" * 65)
print(f"FINE-GRAINED FINGERPRINTING RESULTS (6 classes, {n_splits}-fold CV)")
print("=" * 65)
print(f"{'Variant':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)
for v in variants:
    sub = detail_df[detail_df["variant"] == v]
    print(f"{v:<20} "
          f"{sub['precision'].mean():.3f}±{sub['precision'].std():.3f}  "
          f"{sub['recall'].mean():.3f}±{sub['recall'].std():.3f}  "
          f"{sub['f1'].mean():.3f}±{sub['f1'].std():.3f}")
print("=" * 65)

# ── Per-fold F1 ───────────────────────────────────────────────────────────────
print("\nPer-fold F1 (macro):")
pivot = detail_df.pivot(index="fold", columns="variant", values="f1")
print(pivot[variants].round(3).to_string())

# ── Best variant confusion matrix ─────────────────────────────────────────────
best_variant = detail_df.groupby("variant")["f1"].mean().idxmax()
present_labels = sorted(np.unique(y))
arch_order     = [ARCH_NAMES[i] for i in present_labels]
print(f"\nConfusion matrix — best variant: {best_variant}")
cm = confusion_matrix(all_y_true[best_variant], all_y_pred[best_variant])
cm_df = pd.DataFrame(cm, index=arch_order, columns=arch_order)
print(cm_df.to_string())

# ── Per-class breakdown ───────────────────────────────────────────────────────
print(f"\nPer-class report — {best_variant}:")
print(classification_report(
    all_y_true[best_variant], all_y_pred[best_variant],
    target_names=arch_order, digits=3))

# ── Feature importance ────────────────────────────────────────────────────────
print("Top 15 most important flow features (full dataset RF):")
rf_full = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42, n_jobs=-1)
rf_full.fit(_scale(X_flow), y)
imp_df = pd.DataFrame({
    "feature":    FLOW_COLS,
    "importance": rf_full.feature_importances_,
}).sort_values("importance", ascending=False)
print(imp_df.head(15).to_string(index=False))

# ── CNN-only and RNN-only sub-analysis ───────────────────────────────────────
cnn_present = [l for l in present_labels if l in [0, 1, 2]]
rnn_present = [l for l in present_labels if l in [3, 4, 5]]

cnn_mean, cnn_std, cnn_per_arch = 0.0, 0.0, {}
print("\n--- CNN family only (SimpleCNN vs ResNet18 vs MobileNet) ---")
if len(cnn_present) >= 2:
    cnn_mask = df["label"].isin(cnn_present)
    Xf_cnn = _scale(X_flow[cnn_mask])
    y_cnn  = y[cnn_mask]
    cv_cnn = StratifiedKFold(n_splits=min(5, int(np.unique(y_cnn, return_counts=True)[1].min())), shuffle=True, random_state=42)
    cnn_f1s, all_cnn_true, all_cnn_pred = [], [], []
    for tr, te in cv_cnn.split(Xf_cnn, y_cnn):
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(Xf_cnn[tr], y_cnn[tr])
        pred = rf.predict(Xf_cnn[te])
        cnn_f1s.append(f1_score(y_cnn[te], pred, average="macro"))
        all_cnn_true.extend(y_cnn[te]); all_cnn_pred.extend(pred)
    cnn_mean = float(np.mean(cnn_f1s))
    cnn_std  = float(np.std(cnn_f1s))
    print(f"CNN intra-family F1: {cnn_mean:.3f} ± {cnn_std:.3f}")
    rpt = classification_report(all_cnn_true, all_cnn_pred, output_dict=True, zero_division=0)
    cnn_per_arch = {ARCH_NAMES[l]: round(rpt[str(l)]['f1-score'], 4)
                    for l in cnn_present if str(l) in rpt}
else:
    print(f"Skipped — only {len(cnn_present)} CNN architecture(s) in dataset.")

rnn_mean, rnn_std, rnn_per_arch = 0.0, 0.0, {}
print("\n--- RNN family only (GRU vs LSTM vs BiLSTM) ---")
if len(rnn_present) >= 2:
    rnn_mask = df["label"].isin(rnn_present)
    Xf_rnn = _scale(X_flow[rnn_mask])
    y_rnn  = y[rnn_mask]
    cv_rnn = StratifiedKFold(n_splits=min(5, int(np.unique(y_rnn, return_counts=True)[1].min())), shuffle=True, random_state=42)
    rnn_f1s, all_rnn_true, all_rnn_pred = [], [], []
    for tr, te in cv_rnn.split(Xf_rnn, y_rnn):
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(Xf_rnn[tr], y_rnn[tr])
        pred = rf.predict(Xf_rnn[te])
        rnn_f1s.append(f1_score(y_rnn[te], pred, average="macro"))
        all_rnn_true.extend(y_rnn[te]); all_rnn_pred.extend(pred)
    rnn_mean = float(np.mean(rnn_f1s))
    rnn_std  = float(np.std(rnn_f1s))
    print(f"RNN intra-family F1: {rnn_mean:.3f} ± {rnn_std:.3f}")
    rpt = classification_report(all_rnn_true, all_rnn_pred, output_dict=True, zero_division=0)
    rnn_per_arch = {ARCH_NAMES[l]: round(rpt[str(l)]['f1-score'], 4)
                    for l in rnn_present if str(l) in rpt}
else:
    print(f"Skipped — only {len(rnn_present)} RNN architecture(s) in dataset.")

# ── Write family_f1.json ──────────────────────────────────────────────────────
family_f1 = {
    "cnn": {"mean": round(cnn_mean, 4), "std": round(cnn_std, 4), "per_arch": cnn_per_arch} if len(cnn_present) >= 2 else None,
    "rnn": {"mean": round(rnn_mean, 4), "std": round(rnn_std, 4), "per_arch": rnn_per_arch} if len(rnn_present) >= 2 else None,
}
with open("family_f1.json", "w") as f:
    json.dump(family_f1, f, indent=2)
print(f"Saved: family_f1.json → CNN {cnn_mean:.3f}±{cnn_std:.3f}, RNN {rnn_mean:.3f}±{rnn_std:.3f}")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
detail_df.to_csv("cv_results_multiclass.csv", index=False)
imp_df.to_csv("feature_importance_multiclass.csv", index=False)
cm_df.to_csv("confusion_matrix.csv")
print("\nSaved: cv_results_multiclass.csv, feature_importance_multiclass.csv, confusion_matrix.csv")

# ── Save trained model ────────────────────────────────────────────────────────
print("\n[classify] Training final models on full dataset for deployment...")
os.makedirs("model", exist_ok=True)

fs_flow = StandardScaler().fit(X_flow)
fs_pkt  = StandardScaler().fit(X_pkt)
Xf_all  = np.nan_to_num(fs_flow.transform(X_flow))
Xp_all  = np.nan_to_num(fs_pkt.transform(X_pkt))

fr_flow = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
fr_pkt  = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
fr_flow.fit(Xf_all, y)
fr_pkt.fit(Xp_all, y)

_meta_all = np.hstack([fr_flow.predict_proba(Xf_all), fr_pkt.predict_proba(Xp_all)])
fm_lr  = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000, C=1.0)
fm_xgb = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, random_state=42, max_depth=3)
fm_lr.fit(_meta_all, y)
fm_xgb.fit(_meta_all, y)

_best_f1_mean = float(detail_df[detail_df["variant"] == best_variant]["f1"].mean())
_best_f1_std  = float(detail_df[detail_df["variant"] == best_variant]["f1"].std())

bundle = {
    "rf_flow":      fr_flow,
    "rf_pkt":       fr_pkt,
    "meta_lr":      fm_lr,
    "meta_xgb":     fm_xgb,
    "scaler_flow":  fs_flow,
    "scaler_pkt":   fs_pkt,
    "flow_cols":    FLOW_COLS,
    "pkt_cols":     PKT_COLS,
    "best_variant": best_variant,
    "arch_names":   ARCH_NAMES,
    "trained_at":   datetime.now().isoformat(),
    "cv_f1_mean":   _best_f1_mean,
    "cv_f1_std":    _best_f1_std,
    "n_windows":    len(df),
}
joblib.dump(bundle, "model/model.pkl")
print(f"Saved: model/model.pkl  "
      f"(variant={best_variant}, F1={_best_f1_mean:.3f}±{_best_f1_std:.3f})")
