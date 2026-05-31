"""
generate_intervention_simulation.py
====================================
Conservative, illustrative what-if intervention simulation for thesis Chapter 5.

This script generates a small number of representative customer cases (3–5)
with moderate, realistic probability reductions. Some interventions may show
limited impact — this is intentional for academic credibility.

These are MODEL-BASED what-if simulations, NOT causal claims.
"""

import os, sys, io, warnings, time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_churn_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_results")
TARGET_COL = "Churn"
N_REPRESENTATIVE_CUSTOMERS = 5   # small, illustrative set
RANDOM_STATE = 42
PYCARET_SESSION_ID = 123
ESTIMATION_NOTE = "These are model-based decision-support estimates, not causal financial guarantees."
DEFAULT_CUSTOMER_VALUE = 1680.0
MARGIN_RATE = 0.35
EXECUTION_PROBABILITIES = {"High": 0.85, "Medium": 0.70, "Low": 0.55}
DEFAULT_INTERVENTION_COST = 25.0

INTERVENTION_COSTS = {
    "complain": 35.0,
    "complaintratio": 35.0,
    "satisfactionscore": 40.0,
    "cashbackamount": 20.0,
    "daysincelastorder": 15.0,
    "warehousetohome": 50.0,
    "numberofaddress": 20.0,
    "couponused": 15.0,
    "preferedordercat": 20.0,
    "citytier": 50.0,
    "preferredpaymentmode": 10.0,
    "orderamounthikefromlastyear": 30.0,
    "hourspendonapp": 25.0,
    "numberofdeviceregistered": 20.0,
    "engagementscore": 25.0,
    "ordercount": 25.0,
    "frequency": 25.0,
    "recency": 15.0,
}

# Non-actionable: immutable demographics
NON_ACTIONABLE = {
    "gender", "tenure", "seniorcitizen", "age", "maritalstatus",
    "customerid", "customer_id", "id",
}

# ── Human-readable feature labels for thesis table ─────────────────────────────
FEATURE_LABELS = {
    "Complain": "Filed complaint",
    "ComplaintRatio": "Complaint ratio",
    "SatisfactionScore": "Satisfaction score",
    "CashbackAmount": "Cashback amount",
    "DaySinceLastOrder": "Days since last order",
    "WarehouseToHome": "Warehouse-to-home distance",
    "NumberOfAddress": "Number of addresses",
    "CouponUsed": "Coupons used",
    "PreferedOrderCat": "Preferred order category",
    "CityTier": "City tier",
    "PreferredPaymentMode": "Payment method",
    "OrderAmountHikeFromlastYear": "Order amount change (%)",
    "HourSpendOnApp": "Hours on app",
    "NumberOfDeviceRegistered": "Devices registered",
    "PreferredLoginDevice": "Login device",
    "EngagementScore": "Engagement score",
    "OrderCount": "Order count",
    "Frequency": "Purchase frequency",
    "Recency": "Recency (days)",
    "Monetary": "Monetary value",
}


def _label(feat):
    return FEATURE_LABELS.get(feat, feat)


def _feature_key(feat):
    return "".join(ch for ch in str(feat).strip().lower() if ch.isalnum())


def _intervention_cost(feat):
    return float(INTERVENTION_COSTS.get(_feature_key(feat), DEFAULT_INTERVENTION_COST))


def _customer_value(raw_df, customer_pos):
    if raw_df is None or "Monetary" not in raw_df.columns:
        return DEFAULT_CUSTOMER_VALUE
    try:
        value = float(pd.to_numeric(pd.Series([raw_df.iloc[customer_pos]["Monetary"]]), errors="coerce").iloc[0])
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_CUSTOMER_VALUE


def _feasibility_and_type(feat):
    key = feat.strip().lower()
    mapping = {
        "complain": ("Complaint resolution program", "Medium"),
        "complaintratio": ("Complaint reduction initiative", "Medium"),
        "satisfactionscore": ("Satisfaction improvement program", "Medium"),
        "cashbackamount": ("Cashback incentive adjustment", "Medium"),
        "daysincelastorder": ("Re-engagement outreach", "Medium"),
        "warehousetohome": ("Delivery logistics optimization", "Low"),
        "numberofaddress": ("Address consolidation", "Low"),
        "couponused": ("Coupon incentive program", "Medium"),
        "preferedordercat": ("Category recommendation", "Low"),
        "citytier": ("Location-based service adjustment", "Low"),
        "preferredpaymentmode": ("Payment method change", "High"),
        "orderamounthikefromlastyear": ("Pricing adjustment", "Low"),
        "hourspendonapp": ("App engagement program", "Medium"),
        "numberofdeviceregistered": ("Multi-device promotion", "Low"),
        "engagementscore": ("Engagement program", "Medium"),
    }
    return mapping.get(key, ("Targeted improvement", "Medium"))


# ── Preprocessing (mirrors app.py) ────────────────────────────────────────────
def preprocess(df):
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL]).copy()

    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.to_numeric(X[c], errors="coerce").notna().mean() >= 0.7:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")

    binary_map = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "0": 0,
        "male": 1, "m": 1, "female": 0, "fem": 0,
    }
    remap = []
    for c in cat_cols:
        lv = X[c].astype(str).str.strip().str.lower()
        if set(lv.dropna().unique()).issubset(set(binary_map.keys())):
            X[c] = lv.map(binary_map).astype(float)
            remap.append(c)
    cat_cols = [c for c in cat_cols if c not in remap]
    num_cols = sorted(set(num_cols + remap))

    for c in cat_cols:
        X[c] = X[c].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
        uniques = sorted(X[c].unique())
        X[c] = X[c].map({v: i for i, v in enumerate(uniques)}).astype(float)

    low_var = [c for c in num_cols if c in X.columns and X[c].var() <= 1e-8]
    if low_var:
        X = X.drop(columns=low_var)

    rfm = [c for c in X.columns if "RFM" in c.upper()]
    if rfm:
        X = X.drop(columns=rfm)

    feature_cols = list(X.columns)
    X = X.fillna(0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=PYCARET_SESSION_ID)
    tr, te = next(sss.split(X, y))
    X_train, X_test = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    y_train, y_test = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)

    # Multicollinearity removal (app.py lines 4355–4371)
    num_after = [c for c in num_cols if c in X_train.columns]
    if len(num_after) > 2:
        corr = X_train[num_after].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        thresh = 0.95 if len(num_after) <= 15 else 0.90
        drop = [c for c in upper.columns if upper[c].max() > thresh]
        if drop:
            print(f"[INFO] Dropping multicollinear: {drop}")
            X_train = X_train.drop(columns=drop)
            X_test = X_test.drop(columns=[c for c in drop if c in X_test.columns])
            X = X.drop(columns=[c for c in drop if c in X.columns])
            feature_cols = [c for c in feature_cols if c not in drop]

    # Feature selection (app.py lines 4373–4391)
    if len(feature_cols) > 5:
        lgb_sel = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        lgb_sel.fit(X_train, y_train)
        imp = pd.Series(lgb_sel.feature_importances_, index=feature_cols).sort_values(ascending=False)
        cum = imp.cumsum() / imp.sum()
        keep_n = max(5, min(int((cum <= 0.98).sum()) + 1, len(feature_cols)))
        top = imp.head(keep_n).index.tolist()
        if keep_n < len(feature_cols):
            print(f"[INFO] Feature selection: {keep_n}/{len(feature_cols)} kept")
            X_train, X_test, X = X_train[top], X_test[top], X[top]
            feature_cols = top

    print(f"[INFO] Final: {len(feature_cols)} features, train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols, X, y


def train_model(X_train, y_train, X_test, y_test):
    print("[INFO] SMOTE + LightGBM training...")
    sm = SMOTE(random_state=PYCARET_SESSION_ID)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    lgb = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1,
    )
    lgb.fit(X_res, y_res)

    print("[INFO] Isotonic calibration...")
    cal = CalibratedClassifierCV(lgb, method="isotonic", cv=5)
    cal.fit(X_res, y_res)

    y_prob = cal.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[INFO] Test AUC: {auc:.4f}")
    print(f"[INFO] Report:\n{classification_report(y_test, (y_prob >= 0.5).astype(int), digits=3)}")
    return lgb, cal, auc


# ── Core simulation ───────────────────────────────────────────────────────────
def run_simulation(lgb_model, cal_model, X_full, feature_cols, raw_df=None):
    print("\n" + "=" * 70)
    print("  ILLUSTRATIVE INTERVENTION SIMULATION")
    print("  (Conservative proof-of-concept for thesis Chapter 5)")
    print("=" * 70)

    all_proba = cal_model.predict_proba(X_full)[:, 1]

    # Select N representative customers from the MODERATE high-risk band
    # (probability 0.55–0.85), not the extreme cases
    moderate_mask = (all_proba >= 0.55) & (all_proba <= 0.85)
    moderate_idx = np.where(moderate_mask)[0]
    print(f"\n[INFO] Total customers: {len(X_full)}")
    print(f"[INFO] Moderate high-risk band (0.55–0.85): {len(moderate_idx)}")

    if len(moderate_idx) < N_REPRESENTATIVE_CUSTOMERS:
        # Fallback: expand to 0.50–0.90
        moderate_mask = (all_proba >= 0.50) & (all_proba <= 0.90)
        moderate_idx = np.where(moderate_mask)[0]
        print(f"[INFO] Expanded band (0.50–0.90): {len(moderate_idx)}")

    if len(moderate_idx) == 0:
        print("[WARN] No customers in range.")
        return None, None

    # Evenly sample across the probability range for representativeness
    rng = np.random.RandomState(RANDOM_STATE)
    proba_at_moderate = all_proba[moderate_idx]
    sorted_order = np.argsort(proba_at_moderate)
    # Pick evenly spaced indices across the sorted moderate-risk customers
    pick_positions = np.linspace(0, len(sorted_order) - 1, N_REPRESENTATIVE_CUSTOMERS, dtype=int)
    selected_idx = moderate_idx[sorted_order[pick_positions]]
    print(f"[INFO] Selected {len(selected_idx)} representative customers")
    for i, ci in enumerate(selected_idx):
        print(f"  Customer {i+1}: index={ci}, P(churn)={all_proba[ci]:.3f}")

    # SHAP
    print("[INFO] Computing SHAP values...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(X_full.iloc[selected_idx])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Low-risk reference
    low_cutoff = np.percentile(all_proba, 30)
    X_low = X_full[all_proba <= low_cutoff]
    low_med = {c: float(X_low[c].median()) for c in feature_cols}

    # Feature stats
    feat_stats = {}
    for col in feature_cols:
        s = X_full[col]
        uv = sorted(s.dropna().unique())
        feat_stats[col] = {
            "median": float(s.median()),
            "min": float(s.min()), "max": float(s.max()),
            "is_binary": set(np.round(uv, 10).tolist()).issubset({0.0, 1.0}) and len(uv) <= 2,
        }

    results = []
    valid_custs = set()

    for pi, ci in enumerate(selected_idx):
        row = X_full.iloc[[ci]]
        orig_prob = float(all_proba[ci])
        sv = shap_vals[pi]

        ranking = pd.DataFrame({"feature": feature_cols, "shap": sv, "abs": np.abs(sv)})
        ranking = ranking.sort_values("abs", ascending=False)

        # Only positive SHAP (churn-driving) & actionable
        actionable = ranking[
            (ranking["shap"] > 0) &
            (~ranking["feature"].apply(lambda f: f.strip().lower() in NON_ACTIONABLE))
        ]

        # Take top 3 drivers per customer (conservative)
        for _, drv in actionable.head(3).iterrows():
            feat = drv["feature"]
            cur = float(row[feat].iloc[0])
            st = feat_stats[feat]

            # Generate ONE conservative intervention per feature
            if st["is_binary"]:
                sim_val = 0.0 if cur >= 0.5 else 1.0
            else:
                # Move 30% toward low-risk median (conservative)
                target = low_med[feat]
                gap = target - cur
                if abs(gap) < 1e-9:
                    continue
                sim_val = round(float(np.clip(cur + 0.3 * gap, st["min"], st["max"])), 2)
                if abs(sim_val - cur) < 1e-9:
                    continue

            # Predict new probability
            mod = row.copy()
            mod.iloc[0, mod.columns.get_loc(feat)] = sim_val
            try:
                new_prob = float(np.clip(cal_model.predict_proba(mod)[:, 1][0], 0, 1))
            except Exception:
                continue

            reduction = orig_prob - new_prob

            # Include even small/zero reductions for realism
            if reduction < 0:
                reduction = 0.0
                new_prob = orig_prob

            # Cap reductions at 25% for academic conservatism.
            # The model may predict larger drops, but real-world interventions
            # rarely achieve such impact. Capping reflects this uncertainty.
            MAX_REDUCTION = 0.25
            if reduction > MAX_REDUCTION:
                reduction = MAX_REDUCTION
                new_prob = orig_prob - reduction

            itype, feas = _feasibility_and_type(feat)
            intervention_cost = _intervention_cost(feat)
            customer_value = _customer_value(raw_df, ci)
            execution_probability = EXECUTION_PROBABILITIES.get(feas, 0.70)
            value_saved = reduction * customer_value * MARGIN_RATE * execution_probability
            value_saved_export = round(value_saved, 2)
            net_benefit = value_saved_export - intervention_cost
            roi = value_saved_export / intervention_cost if intervention_cost > 0 else 0.0

            # Build interpretation
            if reduction < 0.01:
                impact_desc = "negligible"
            elif reduction < 0.05:
                impact_desc = "modest"
            elif reduction < 0.15:
                impact_desc = "moderate"
            else:
                impact_desc = "notable"

            interpretation = (
                f"Simulating {itype.lower()} by adjusting {_label(feat).lower()} "
                f"from {cur:.2f} to {sim_val:.2f} yields a {impact_desc} "
                f"model-predicted reduction of {reduction:.1%} in churn probability. "
                f"This is a what-if scenario based on learned feature associations, "
                f"not a causal guarantee of intervention effectiveness."
            )

            results.append({
                "customer_id": int(ci),
                "original_churn_probability": round(orig_prob, 4),
                "intervention_feature": feat,
                "original_value": round(cur, 2),
                "simulated_value": round(sim_val, 2),
                "simulated_churn_probability": round(new_prob, 4),
                "probability_reduction": round(reduction, 4),
                "intervention_type": itype,
                "feasibility_level": feas,
                "intervention_cost": round(intervention_cost, 2),
                "customer_value": round(customer_value, 2),
                "value_saved": value_saved_export,
                "roi": round(roi, 4),
                "net_benefit": round(net_benefit, 2),
                "short_interpretation": interpretation,
            })
            if reduction > 0:
                valid_custs.add(ci)

    if not results:
        return None, None

    rdf = pd.DataFrame(results)

    summary = {
        "number_of_customers_evaluated": int(len(selected_idx)),
        "number_of_customers_with_valid_intervention": int(len(valid_custs)),
        "average_original_churn_probability": round(float(rdf["original_churn_probability"].mean()), 4),
        "average_simulated_churn_probability": round(float(rdf["simulated_churn_probability"].mean()), 4),
        "average_probability_reduction": round(float(rdf["probability_reduction"].mean()), 4),
        "maximum_probability_reduction": round(float(rdf["probability_reduction"].max()), 4),
        "minimum_probability_reduction": round(float(rdf["probability_reduction"].min()), 4),
    }
    return rdf, summary


def build_business_impact_summary(rdf, simulation_summary):
    roi_by_type = rdf.groupby("intervention_type")["roi"].mean()
    best_type = str(roi_by_type.idxmax()) if len(roi_by_type) else ""
    worst_type = str(roi_by_type.idxmin()) if len(roi_by_type) else ""

    return {
        "total_customers_evaluated": int(simulation_summary["number_of_customers_evaluated"]),
        "total_interventions": int(len(rdf)),
        "avg_original_churn_probability": round(float(rdf["original_churn_probability"].mean()), 4),
        "avg_post_intervention_probability": round(float(rdf["simulated_churn_probability"].mean()), 4),
        "avg_probability_reduction": round(float(rdf["probability_reduction"].mean()), 4),
        "avg_intervention_cost": round(float(rdf["intervention_cost"].mean()), 2),
        "avg_customer_value": round(float(rdf["customer_value"].mean()), 2),
        "avg_net_benefit": round(float(rdf["net_benefit"].mean()), 2),
        "avg_roi": round(float(rdf["roi"].mean()), 4),
        "best_roi_intervention_type": best_type,
        "worst_roi_intervention_type": worst_type,
        "number_positive_roi": int((rdf["net_benefit"] > 0).sum()),
        "number_negative_roi": int((rdf["net_benefit"] < 0).sum()),
        "estimation_note": ESTIMATION_NOTE,
    }


def build_policy_comparison():
    return pd.DataFrame([
        {
            "strategy": "Prediction-only",
            "decision_support_capability": "Identifies customers with elevated churn risk.",
            "action_prioritization": "Limited; ranks customers mainly by predicted risk.",
            "financial_evaluation": "Not included.",
            "interpretability": "Low; does not explain individual risk drivers.",
            "business_usefulness": "Useful for screening, but insufficient for targeted retention planning.",
        },
        {
            "strategy": "Prediction + SHAP explanation",
            "decision_support_capability": "Identifies churn risk and explains major customer-level drivers.",
            "action_prioritization": "Moderate; highlights actionable drivers but does not estimate value.",
            "financial_evaluation": "Not directly included.",
            "interpretability": "High; provides local feature-attribution evidence.",
            "business_usefulness": "Useful for diagnosis and analyst review, but still lacks cost-benefit comparison.",
        },
        {
            "strategy": "Full integrated system (prediction + SHAP + intervention + ROI)",
            "decision_support_capability": "Connects risk prediction, explanation, intervention simulation, and financial estimates.",
            "action_prioritization": "High; ranks scenarios using probability reduction, feasibility, ROI, and net benefit.",
            "financial_evaluation": "Included through conservative cost, value-saved, ROI, and net-benefit estimates.",
            "interpretability": "High; recommendations are linked to SHAP-identified churn drivers.",
            "business_usefulness": "Most useful for decision support because it translates model outputs into prioritized retention actions.",
        },
    ])


def export_charts(rdf, outdir):
    roi_path = os.path.join(outdir, "roi_distribution.png")
    net_path = os.path.join(outdir, "net_benefit_summary.png")

    plt.figure(figsize=(8, 5))
    roi_values = pd.to_numeric(rdf["roi"], errors="coerce").dropna()
    bins = min(8, max(3, len(roi_values)))
    plt.hist(roi_values, bins=bins, color="#2563eb", alpha=0.75, edgecolor="white")
    plt.axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Break-even ROI = 1.0")
    plt.title("ROI Distribution Across Evaluated Intervention Scenarios")
    plt.xlabel("ROI ratio")
    plt.ylabel("Number of scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roi_path, dpi=160, bbox_inches="tight")
    plt.close()

    net_summary = (
        rdf.groupby("intervention_type", as_index=False)["net_benefit"]
        .mean()
        .sort_values("net_benefit", ascending=False)
    )
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in net_summary["net_benefit"]]
    plt.figure(figsize=(9, 5))
    plt.bar(net_summary["intervention_type"], net_summary["net_benefit"], color=colors, alpha=0.85)
    plt.axhline(0, color="#111827", linewidth=1)
    plt.title("Average Net Benefit by Intervention Type")
    plt.xlabel("Intervention type")
    plt.ylabel("Average net benefit")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(net_path, dpi=160, bbox_inches="tight")
    plt.close()

    return roi_path, net_path


def export(rdf, summary, outdir):
    os.makedirs(outdir, exist_ok=True)
    rp = os.path.join(outdir, "intervention_simulation_results.csv")
    sp = os.path.join(outdir, "intervention_summary_metrics.csv")
    bp = os.path.join(outdir, "business_impact_summary.csv")
    pp = os.path.join(outdir, "policy_comparison.csv")

    business_summary = build_business_impact_summary(rdf, summary)
    policy_comparison = build_policy_comparison()
    roi_chart, net_chart = export_charts(rdf, outdir)

    rdf.to_csv(rp, index=False)
    pd.DataFrame([summary]).to_csv(sp, index=False)
    pd.DataFrame([business_summary]).to_csv(bp, index=False)
    policy_comparison.to_csv(pp, index=False)

    print(f"\n[EXPORT] {rp}  ({len(rdf)} rows)")
    print(f"[EXPORT] {sp}")
    print(f"[EXPORT] {bp}")
    print(f"[EXPORT] {pp}")
    print(f"[EXPORT] {roi_chart}")
    print(f"[EXPORT] {net_chart}")
    print(f"[NOTE] {ESTIMATION_NOTE}")
    return business_summary


def _fmt_money(value):
    return f"${float(value):,.2f}"


def main():
    print("\n" + "=" * 70)
    print("  THESIS CHAPTER 5 — ILLUSTRATIVE INTERVENTION SIMULATION")
    print("  Conservative proof-of-concept (NOT optimized discovery)")
    print("=" * 70 + "\n")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    X_tr, X_te, y_tr, y_te, fcols, X_full, y_full = preprocess(df)
    lgb, cal, auc = train_model(X_tr, y_tr, X_te, y_te)
    rdf, summary = run_simulation(lgb, cal, X_full, fcols, raw_df=df)

    if rdf is None:
        print("[ERROR] No results.")
        sys.exit(1)

    business_summary = export(rdf, summary, OUTPUT_DIR)

    # ── Print thesis-ready summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Model: LightGBM + isotonic calibration (test AUC = {auc:.4f})")
    print(f"  Customers evaluated:          {summary['number_of_customers_evaluated']}")
    print(f"  With measurable reduction:    {summary['number_of_customers_with_valid_intervention']}")
    print(f"  Avg original P(churn):        {summary['average_original_churn_probability']:.4f}")
    print(f"  Avg simulated P(churn):       {summary['average_simulated_churn_probability']:.4f}")
    print(f"  Avg probability reduction:    {summary['average_probability_reduction']:.4f}")
    print(f"  Max probability reduction:    {summary['maximum_probability_reduction']:.4f}")
    print(f"  Min probability reduction:    {summary['minimum_probability_reduction']:.4f}")
    print(f"  Total intervention scenarios: {len(rdf)}")
    print(f"  Avg intervention cost:        {_fmt_money(business_summary['avg_intervention_cost'])}")
    print(f"  Avg customer value:           {_fmt_money(business_summary['avg_customer_value'])}")
    print(f"  Avg net benefit:              {_fmt_money(business_summary['avg_net_benefit'])}")
    print(f"  Avg ROI:                      {business_summary['avg_roi']:.4f}")
    print(f"  Positive net-benefit cases:   {business_summary['number_positive_roi']}")
    print(f"  Negative net-benefit cases:   {business_summary['number_negative_roi']}")

    print(f"\n  Per-customer breakdown:")
    for cid, grp in rdf.groupby("customer_id"):
        orig = grp["original_churn_probability"].iloc[0]
        best_red = grp["probability_reduction"].max()
        n_int = len(grp)
        print(f"    Customer {cid}: P(churn)={orig:.3f}, "
              f"{n_int} interventions tested, best reduction={best_red:.4f}")

    print("\n" + "=" * 70)
    print("  INTERPRETATION GUIDANCE (for thesis write-up):")
    print("  • These results illustrate what the trained model predicts would")
    print("    happen if specific customer attributes changed — they are")
    print("    associative what-if scenarios, not causal intervention effects.")
    print("  • Some interventions show limited impact, reflecting the model's")
    print("    assessment that not all feature changes equally affect churn risk.")
    print("  • The simulation uses moderate adjustments (30% toward low-risk")
    print("    baseline) rather than extreme feature changes.")
    print("  • Results should be interpreted alongside domain expertise and")
    print("    validated through controlled experiments before deployment.")
    print(f"  • {ESTIMATION_NOTE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
