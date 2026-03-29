import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")

META_COLS  = ["label", "pcap", "window", "n_packets"]
FLOW_COLS  = [c for c in df.columns if c.startswith(("pkt_rate","up_down","up_","down_","iat_"))]
PKT_COLS   = [c for c in df.columns if c.startswith(("hist_","first_","last_"))]

X_flow = df[FLOW_COLS].values.astype(np.float32)
X_pkt  = df[PKT_COLS].values.astype(np.float32)
y      = df["label"].values

print(f"Dataset: {len(df)} windows | {len(FLOW_COLS)} flow features | {len(PKT_COLS)} packet features")
print(f"Classes: {np.bincount(y)} (0=CNN, 1=RNN)\n")

# ── Cross-validation ──────────────────────────────────────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()

results = {
    "flow_only":   {"MetaLR": [], "MetaXGB": []},
    "packet_only": {"MetaLR": [], "MetaXGB": []},
    "fusion":      {"MetaLR": [], "MetaXGB": []},
}

fold_details = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_flow, y)):
    Xf_tr, Xf_te = X_flow[train_idx], X_flow[test_idx]
    Xp_tr, Xp_te = X_pkt[train_idx],  X_pkt[test_idx]
    y_tr,  y_te  = y[train_idx],       y[test_idx]

    # Scale
    Xf_tr = scaler.fit_transform(Xf_tr)
    Xf_te = scaler.transform(Xf_te)
    Xp_tr_s = StandardScaler().fit_transform(Xp_tr)
    Xp_te_s = StandardScaler().fit_transform(Xp_te)

    # ── Base models (Random Forest) ───────────────────────────────────────────
    rf_flow = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_pkt = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )

    rf_flow.fit(Xf_tr, y_tr)
    rf_pkt.fit(Xp_tr_s, y_tr)

    # Probability outputs for meta-features
    pf_flow_tr = rf_flow.predict_proba(Xf_tr)
    pf_flow_te = rf_flow.predict_proba(Xf_te)
    pf_pkt_tr  = rf_pkt.predict_proba(Xp_tr_s)
    pf_pkt_te  = rf_pkt.predict_proba(Xp_te_s)

    # ── Meta-feature vectors ──────────────────────────────────────────────────
    meta_tr = np.hstack([pf_flow_tr, pf_pkt_tr])
    meta_te = np.hstack([pf_flow_te, pf_pkt_te])

    # ── Meta-classifiers ─────────────────────────────────────────────────────
    meta_lr  = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    meta_xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    meta_lr.fit(meta_tr, y_tr)
    meta_xgb.fit(meta_tr, y_tr)

    # ── Evaluate all combinations ─────────────────────────────────────────────
    def score(y_true, y_pred, tag):
        return {
            "fold":      fold + 1,
            "variant":   tag,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }

    fold_details += [
        score(y_te, rf_flow.predict(Xf_te),        "flow_only"),
        score(y_te, rf_pkt.predict(Xp_te_s),       "packet_only"),
        score(y_te, meta_lr.predict(meta_te),       "fusion_MetaLR"),
        score(y_te, meta_xgb.predict(meta_te),      "fusion_MetaXGB"),
    ]

# ── Summary table ─────────────────────────────────────────────────────────────
detail_df = pd.DataFrame(fold_details)

print("=" * 65)
print("FLARE FINGERPRINTING RESULTS (5-fold CV)")
print("=" * 65)
print(f"{'Variant':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)

for variant in ["flow_only", "packet_only", "fusion_MetaLR", "fusion_MetaXGB"]:
    sub = detail_df[detail_df["variant"] == variant]
    p   = sub["precision"].mean()
    r   = sub["recall"].mean()
    f1  = sub["f1"].mean()
    ps  = sub["precision"].std()
    rs  = sub["recall"].std()
    f1s = sub["f1"].std()
    print(f"{variant:<20} {p:.3f}±{ps:.3f}  {r:.3f}±{rs:.3f}  {f1:.3f}±{f1s:.3f}")

print("=" * 65)

# ── Per-fold breakdown ────────────────────────────────────────────────────────
print("\nPer-fold F1 scores:")
pivot = detail_df.pivot(index="fold", columns="variant", values="f1")
print(pivot.round(3).to_string())

# ── Feature importance (flow features) ───────────────────────────────────────
print("\nTop 10 most important flow features:")
rf_full = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_full.fit(scaler.fit_transform(X_flow), y)
importance_df = pd.DataFrame({
    "feature":    FLOW_COLS,
    "importance": rf_full.feature_importances_,
}).sort_values("importance", ascending=False)
print(importance_df.head(10).to_string(index=False))

# ── Save results ──────────────────────────────────────────────────────────────
detail_df.to_csv("cv_results.csv", index=False)
importance_df.to_csv("feature_importance.csv", index=False)
print("\nResults saved to cv_results.csv and feature_importance.csv")
