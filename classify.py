import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")

ARCH_NAMES = {0:"SimpleCNN", 1:"ResNet18", 2:"MobileNet",
              3:"GRU", 4:"LSTM", 5:"BiLSTM"}

FLOW_COLS = [c for c in df.columns if c.startswith(
    ("pkt_rate","up_down","up_","down_","iat_"))]
PKT_COLS  = [c for c in df.columns if c.startswith(
    ("hist_","first_","last_"))]

X_flow = df[FLOW_COLS].values.astype(np.float32)
X_pkt  = df[PKT_COLS].values.astype(np.float32)
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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

variants = ["flow_only", "packet_only", "fusion_MetaLR", "fusion_MetaXGB"]
fold_records = []
all_y_true, all_y_pred = {v: [] for v in variants}, {v: [] for v in variants}

for fold, (train_idx, test_idx) in enumerate(cv.split(X_flow, y)):
    Xf_tr = StandardScaler().fit_transform(X_flow[train_idx])
    Xf_te = StandardScaler().fit_transform(X_flow[test_idx])
    Xp_tr = StandardScaler().fit_transform(X_pkt[train_idx])
    Xp_te = StandardScaler().fit_transform(X_pkt[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    # Base Random Forests
    rf_flow = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_pkt  = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_flow.fit(Xf_tr, y_tr)
    rf_pkt.fit(Xp_tr, y_tr)

    # Meta features
    meta_tr = np.hstack([rf_flow.predict_proba(Xf_tr), rf_pkt.predict_proba(Xp_tr)])
    meta_te = np.hstack([rf_flow.predict_proba(Xf_te), rf_pkt.predict_proba(Xp_te)])

    # Meta classifiers
    meta_lr  = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000, C=1.0)
    meta_xgb = GradientBoostingClassifier(
        n_estimators=200, random_state=42, max_depth=3)
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
print("FINE-GRAINED FINGERPRINTING RESULTS (6 classes, 5-fold CV)")
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
print(f"\nConfusion matrix — best variant: {best_variant}")
cm = confusion_matrix(all_y_true[best_variant], all_y_pred[best_variant])
arch_order = [ARCH_NAMES[i] for i in range(6)]
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
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
scaler = StandardScaler()
rf_full.fit(scaler.fit_transform(X_flow), y)
imp_df = pd.DataFrame({
    "feature":    FLOW_COLS,
    "importance": rf_full.feature_importances_,
}).sort_values("importance", ascending=False)
print(imp_df.head(15).to_string(index=False))

# ── CNN-only and RNN-only sub-analysis ───────────────────────────────────────
print("\n--- CNN family only (SimpleCNN vs ResNet18 vs MobileNet) ---")
cnn_mask = df["label"].isin([0,1,2])
Xf_cnn = StandardScaler().fit_transform(X_flow[cnn_mask])
y_cnn  = y[cnn_mask]
cv_cnn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnn_f1s = []
for tr, te in cv_cnn.split(Xf_cnn, y_cnn):
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xf_cnn[tr], y_cnn[tr])
    cnn_f1s.append(f1_score(y_cnn[te], rf.predict(Xf_cnn[te]), average="macro"))
print(f"CNN intra-family F1: {np.mean(cnn_f1s):.3f} ± {np.std(cnn_f1s):.3f}")

print("\n--- RNN family only (GRU vs LSTM vs BiLSTM) ---")
rnn_mask = df["label"].isin([3,4,5])
Xf_rnn = StandardScaler().fit_transform(X_flow[rnn_mask])
y_rnn  = y[rnn_mask]
cv_rnn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rnn_f1s = []
for tr, te in cv_rnn.split(Xf_rnn, y_rnn):
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xf_rnn[tr], y_rnn[tr])
    rnn_f1s.append(f1_score(y_rnn[te], rf.predict(Xf_rnn[te]), average="macro"))
print(f"RNN intra-family F1: {np.mean(rnn_f1s):.3f} ± {np.std(rnn_f1s):.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
detail_df.to_csv("cv_results_multiclass.csv", index=False)
imp_df.to_csv("feature_importance_multiclass.csv", index=False)
cm_df.to_csv("confusion_matrix.csv")
print("\nSaved: cv_results_multiclass.csv, feature_importance_multiclass.csv, confusion_matrix.csv")
