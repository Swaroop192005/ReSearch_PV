"""
uncertainty_ood.py -- is the cross-site model overconfident, and can it know it?

(1) CONFORMAL COVERAGE: build 90% prediction intervals calibrated on the source
    site (split conformal). If they cover ~90% within-site but far less on the
    target, the deployed model is OVERCONFIDENT out-of-distribution.
(2) OOD DETECTION: a domain classifier (source vs target) gives an OOD score per
    sample; its AUC measures how distinguishable the target is.
(3) SELECTIVE PREDICTION: abstaining on the most-OOD target samples should lower
    error on the rest (risk-coverage curve) -- i.e., the OOD score is actionable.

Usage:
  python uncertainty_ood.py --source Trina_monoSi --target NIST_Ground_monoSi
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config, data as d, models as m, metrics as met
warnings.filterwarnings("ignore")
FIG = config.ROOT / "figures"; FIG.mkdir(exist_ok=True)

def pick():
    zoo = m.get_regressors()
    for n in ["CatBoost","XGBoost","HistGradientBoosting","GradientBoosting","RandomForest"]:
        if n in zoo: return n, zoo[n]
    n = next(iter(zoo)); return n, zoo[n]

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Trina_monoSi"); ap.add_argument("--target", default="NIST_Ground_monoSi")
    ap.add_argument("--weather", action="store_true"); ap.add_argument("--alpha", type=float, default=0.90)
    args = ap.parse_args()
    df = d.load_data(); name, base = pick()
    Xs, ys = d.get_xy(d.site_frame(df, [args.source]), "power", args.weather)
    Xt, yt = d.get_xy(d.site_frame(df, [args.target]), "power", args.weather)
    rs = config.RANDOM_STATE
    Xtr, Xtmp, ytr, ytmp = train_test_split(Xs, ys, test_size=0.4, random_state=rs)
    Xcal, Xste, ycal, yste = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=rs)
    model = clone(base).fit(Xtr, ytr)

    # (1) split-conformal interval half-width from source calibration residuals
    q = np.quantile(np.abs(ycal - model.predict(Xcal)), args.alpha)
    cov_in  = float(np.mean(np.abs(yste - model.predict(Xste)) <= q))
    cov_out = float(np.mean(np.abs(yt   - model.predict(Xt))   <= q))
    print(f"\n=== CONFORMAL COVERAGE (nominal {args.alpha:.0%}, half-width={q:.4f} CF) | {args.source}->{args.target} ===")
    print(f"  within-site (source) coverage : {cov_in:.1%}   (should be ~{args.alpha:.0%})")
    print(f"  cross-site  (target) coverage : {cov_out:.1%}   <- overconfidence if << {args.alpha:.0%}")

    # (2) OOD separability: domain classifier source vs target
    Xd = pd.concat([Xs, Xt], ignore_index=True); yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    Xda, Xdb, yda, ydb = train_test_split(Xd, yd, test_size=0.4, random_state=rs, stratify=yd)
    dom = HistGradientBoostingClassifier(random_state=rs).fit(Xda, yda)
    ood_auc = roc_auc_score(ydb, dom.predict_proba(Xdb)[:, 1])
    print(f"\n=== OOD DETECTION ===")
    print(f"  source-vs-target separability AUC = {ood_auc:.3f}   (1.0 = trivially distinguishable)")

    # (3) selective prediction on target using OOD score (P(target|x))
    ood_t = dom.predict_proba(Xt)[:, 1]
    err_t = np.abs(yt - model.predict(Xt))
    corr = float(np.corrcoef(ood_t, err_t)[0, 1])
    print(f"  corr(OOD score, |error|) on target = {corr:+.3f}")
    order = np.argsort(ood_t)  # most source-like first
    covs = [0.2,0.4,0.6,0.8,1.0]; rc=[]
    print(f"\n=== SELECTIVE PREDICTION (keep least-OOD fraction) ===")
    print(f"  {'coverage':>9s} {'RMSE_kept':>10s}")
    for c in covs:
        k = max(1,int(c*len(order))); sel = order[:k]
        r = rmse(yt.iloc[sel] if hasattr(yt,'iloc') else yt[sel], model.predict(Xt.iloc[sel]))
        print(f"  {c:9.0%} {r:10.4f}"); rc.append(r)

    # figures
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    ax.bar(["within-site\n(source)","cross-site\n(target)"],[cov_in*100,cov_out*100],color=["#2E7D32","#E67E22"])
    ax.axhline(args.alpha*100,color="#1F4E79",ls="--",lw=1.3,label=f"nominal {args.alpha:.0%}")
    ax.set_ylabel("Interval coverage (%)"); ax.set_ylim(0,105)
    ax.set_title("Cross-site intervals are overconfident"); ax.legend()
    for i,v in enumerate([cov_in*100,cov_out*100]): ax.text(i,v+1,f"{v:.0f}%",ha="center")
    fig.tight_layout(); fig.savefig(FIG/"unc_coverage.png",dpi=150); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,4.2))
    ax.plot([c*100 for c in covs], rc, "o-", color="#1F4E79")
    ax.set_xlabel("Coverage: least-OOD target samples kept (%)"); ax.set_ylabel("RMSE on kept samples (CF)")
    ax.set_title("Flagging OOD samples improves reliability (risk–coverage)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(FIG/"unc_riskcoverage.png",dpi=150); plt.close(fig)
    print(f"\n[done] figures: figures/unc_coverage.png, figures/unc_riskcoverage.png")

if __name__ == "__main__": main()
