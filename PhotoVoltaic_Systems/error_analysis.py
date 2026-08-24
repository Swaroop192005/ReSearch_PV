"""
error_analysis.py -- where and why does cross-climate transfer fail?

Trains a model on the source site and predicts on the target, then decomposes
the error against a within-site (out-of-fold) reference:
  * RMSE and bias by irradiance level (clear midday vs low/variable light)
  * RMSE by season
  * a calibration fit (actual vs predicted) revealing systematic bias
It prints summary tables and saves diagnostic figures to figures/.

Usage:
  python error_analysis.py --source Trina_monoSi --target NIST_Ground_monoSi
  python error_analysis.py --source Trina_monoSi --target NIST_Ground_monoSi --weather
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, data as d, models as m
warnings.filterwarnings("ignore")

FIG = config.ROOT / "figures"; FIG.mkdir(exist_ok=True)
SEASON = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
          6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}

def pick():
    zoo = m.get_regressors()
    for n in ["CatBoost","XGBoost","HistGradientBoosting","GradientBoosting","RandomForest"]:
        if n in zoo: return n, zoo[n]
    n = next(iter(zoo)); return n, zoo[n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Trina_monoSi")
    ap.add_argument("--target", default="NIST_Ground_monoSi")
    ap.add_argument("--weather", action="store_true")
    args = ap.parse_args()
    df = d.load_data(); name, base = pick()
    src = d.site_frame(df, [args.source]); tgt = d.site_frame(df, [args.target]).reset_index(drop=True)
    Xs, ys = d.get_xy(src, "power", args.weather)
    Xt, yt = d.get_xy(tgt, "power", args.weather)
    cross_pred = clone(base).fit(Xs, ys).predict(Xt)
    within_pred = cross_val_predict(clone(base), Xt, yt, cv=5)
    tgt["actual"], tgt["cross_pred"], tgt["within_pred"] = yt, cross_pred, within_pred
    tgt["cross_res"] = tgt["cross_pred"] - tgt["actual"]
    tgt["within_res"] = tgt["within_pred"] - tgt["actual"]

    print(f"\n=== ERROR BY IRRADIANCE  |  {args.source} -> {args.target}  |  model={name} ===")
    bins=[0,200,400,600,800,1000,3000]; labels=["0-200","200-400","400-600","600-800","800-1000",">1000"]
    tgt["irr_bin"]=pd.cut(tgt["irradiance"], bins=bins, labels=labels, include_lowest=True)
    print(f"{'irr(W/m2)':11s} {'n':>6s} {'RMSE_within':>12s} {'RMSE_cross':>11s} {'bias_cross':>11s}")
    ib=[]
    for lab in labels:
        g=tgt[tgt.irr_bin==lab]
        if len(g)==0: continue
        rw=np.sqrt((g.within_res**2).mean()); rc=np.sqrt((g.cross_res**2).mean()); bc=g.cross_res.mean()
        print(f"{lab:11s} {len(g):6d} {rw:12.4f} {rc:11.4f} {bc:+11.4f}")
        ib.append((lab,rw,rc,bc))

    if "month" in tgt.columns:
        print(f"\n=== ERROR BY SEASON (target, N. hemisphere) ===")
        tgt["season"]=tgt["month"].map(SEASON)
        print(f"{'season':8s} {'n':>6s} {'RMSE_within':>12s} {'RMSE_cross':>11s}")
        for s in ["Winter","Spring","Summer","Autumn"]:
            g=tgt[tgt.season==s]
            if len(g)==0: continue
            print(f"{s:8s} {len(g):6d} {np.sqrt((g.within_res**2).mean()):12.4f} {np.sqrt((g.cross_res**2).mean()):11.4f}")

    a,b0=np.polyfit(tgt.cross_pred, tgt.actual, 1)
    ov=(tgt.cross_res>0).mean()*100
    print(f"\n=== CALIBRATION (cross) ===")
    print(f"  actual = {a:.3f} * pred {b0:+.3f}   (ideal: slope 1, intercept 0)")
    print(f"  mean cross bias = {tgt.cross_res.mean():+.4f} CF; over-predicts {ov:.0f}% of the time")

    # ---- figures ----
    labs=[x[0] for x in ib]; rw=[x[1] for x in ib]; rc=[x[2] for x in ib]
    xx=np.arange(len(labs)); w=0.4
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.bar(xx-w/2,rw,w,label="within-site",color="#2E7D32")
    ax.bar(xx+w/2,rc,w,label="cross-climate",color="#E67E22")
    ax.set_xticks(xx); ax.set_xticklabels(labs); ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel("RMSE (capacity factor)"); ax.set_title("Cross-climate error concentrates at high irradiance")
    ax.legend(); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(FIG/"err_by_irradiance.png",dpi=150); plt.close(fig)

    fig,ax=plt.subplots(figsize=(5.2,5))
    s=tgt.sample(min(3000,len(tgt)),random_state=0)
    ax.scatter(s.cross_pred,s.actual,s=4,alpha=0.25,color="#1F4E79")
    lim=[0,max(tgt.actual.max(),tgt.cross_pred.max())*1.02]
    ax.plot(lim,lim,"k--",lw=1,label="ideal (y=x)")
    ax.plot(lim,[a*z+b0 for z in lim],color="#E67E22",lw=1.5,label=f"fit: {a:.2f}x{b0:+.2f}")
    ax.set_xlabel("Predicted CF (Alice model)"); ax.set_ylabel("Actual CF (NIST)")
    ax.set_title("Cross-climate calibration"); ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
    fig.tight_layout(); fig.savefig(FIG/"err_calibration.png",dpi=150); plt.close(fig)
    print(f"\n[done] figures: figures/err_by_irradiance.png, figures/err_calibration.png")

if __name__ == "__main__":
    main()
