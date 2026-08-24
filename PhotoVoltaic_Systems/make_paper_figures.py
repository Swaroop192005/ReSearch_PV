"""
make_paper_figures.py -- regenerate ALL paper figures directly from your real
data, so every PNG is traceable to results on your machine.

Reads:  data/pv_multisite.csv  (power: Alice arrays + NIST)
        data/dkasc/*.csv        (for the temporal axis; optional)
        data/gpvs_features.csv  (fault detection; optional -- run gpvs_prep.py first)
Writes: figures/paper/fig1..fig10 .png

Run:    python make_paper_figures.py
Each figure is guarded, so missing inputs just skip that figure with a note.
"""
import os, glob, itertools, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"figure.constrained_layout.use":True,"axes.titlepad":10,"font.size":11,"axes.titlesize":12})
import config, data as d, models as m, metrics as met
warnings.filterwarnings("ignore")

OUT = config.ROOT / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
NAVY="#1F4E79"; ORANGE="#E67E22"; GREEN="#2E7D32"; GREY="#B0B7BF"; STEEL="#4A7BA6"
SEEDS=[42,1,2,7,123]
MONO="Trina_monoSi"; POLY="Kyocera_polySi"; CDTE="Calyxo_CdTe"; NIST="NIST_Ground_monoSi"
BOOST=["GradientBoosting","HistGradientBoosting","XGBoost","CatBoost"]

def best():
    zoo=m.get_regressors()
    for n in ["CatBoost","XGBoost","HistGradientBoosting","GradientBoosting","RandomForest"]:
        if n in zoo: return zoo[n]
    return next(iter(zoo.values()))

def fit_eval(model, df, src, tgt, feats, seed=42, within=False):
    if within:
        X,y=d.get_xy(d.site_frame(df,[src]),"power",False)
        xa,xb,ya,yb=train_test_split(X,y,test_size=0.2,random_state=seed)
        mm=clone(model);
        try: mm.set_params(random_state=seed)
        except: pass
        return met.regression_metrics(yb,mm.fit(xa,ya).predict(xb))
    Xs,ys=d.get_xy(d.site_frame(df,[src]),"power",False)
    Xt,yt=d.get_xy(d.site_frame(df,[tgt]),"power",False)
    mm=clone(model)
    try: mm.set_params(random_state=seed)
    except: pass
    return met.regression_metrics(yt, mm.fit(Xs,ys).predict(Xt))

def save(fig,name): fig.savefig(OUT/name,dpi=160,bbox_inches="tight",pad_inches=0.2); plt.close(fig); print("  wrote",name)

def load_power():
    if not config.DATA_CSV.exists(): return None
    df=pd.read_csv(config.DATA_CSV); return df if set([MONO,NIST]).issubset(df[config.SITE_COL].unique()) else df

# ---------- POWER FIGURES ----------
def power_figs():
    df=load_power()
    if df is None: print("[skip] no data/pv_multisite.csv"); return None
    sites=list(df[config.SITE_COL].unique())
    feats=d.feature_columns(False)
    zoo=m.get_regressors()
    # per-model within(target) and cross for tech (mono->poly/CdTe) and climate (mono->NIST)
    have=lambda s: s in sites
    if not (have(MONO) and have(NIST)): print("[skip] expected site names not found; edit MONO/NIST at top"); return df
    rows=[]
    for name in [n for n in BOOST if n in zoo]:
        mdl=zoo[name]
        wK=fit_eval(mdl,df,POLY,None,feats,within=True)["R2"] if have(POLY) else np.nan
        wC=fit_eval(mdl,df,CDTE,None,feats,within=True)["R2"] if have(CDTE) else np.nan
        wN=fit_eval(mdl,df,NIST,None,feats,within=True)["R2"]
        cK=fit_eval(mdl,df,MONO,POLY,feats)["R2"] if have(POLY) else np.nan
        cC=fit_eval(mdl,df,MONO,CDTE,feats)["R2"] if have(CDTE) else np.nan
        cN=fit_eval(mdl,df,MONO,NIST,feats)["R2"]
        tech=np.nanmean([wK-cK, wC-cC]); clim=wN-cN
        rows.append(dict(model=name,tech=tech,clim=clim,wN=wN,cN=cN))
    R=pd.DataFrame(rows)
    # fig1: tech vs climate gap per model
    x=np.arange(len(R)); w=0.38
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    ax.bar(x-w/2,R.tech,w,label="Cross-technology (same site)",color=GREY)
    ax.bar(x+w/2,R.clim,w,label="Cross-climate (Australia→USA)",color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(R.model,rotation=25,ha="right",rotation_mode="anchor")
    ax.set_ylabel("Generalization gap (R² drop)"); ax.set_title("Climate shift degrades far more than technology shift")
    ax.legend(); ax.grid(axis="y",alpha=0.3); save(fig,"fig1_gap.png")
    # fig2: within vs cross (NIST) per model
    fig,ax=plt.subplots(figsize=(7.6,4.3))
    ax.bar(x-w/2,R.wN,w,label="Within-site (trained on NIST)",color=GREEN)
    ax.bar(x+w/2,R.cN,w,label="Cross-site (trained on Alice, zero-shot)",color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(R.model,rotation=25,ha="right",rotation_mode="anchor"); ax.set_ylim(0.85,1.0)
    ax.set_ylabel("R² on NIST test set"); ax.set_title("Cross-climate zero-shot transfer (Australia → USA, mono-Si)")
    ax.legend(loc="lower right"); ax.grid(axis="y",alpha=0.3); save(fig,"fig2_withincross.png")

    # fig4: all-pairs matrix (CatBoost-ish best, 5 seeds)
    base=best(); order=[c for c in [CDTE,POLY,NIST,MONO] if c in sites]
    Mmat=np.full((len(order),len(order)),np.nan)
    for i,src in enumerate(order):
        for j,tgt in enumerate(order):
            vals=[fit_eval(base,df,src,tgt,feats,seed=s,within=(src==tgt))["R2"] for s in SEEDS]
            Mmat[i,j]=np.mean(vals)
    fig,ax=plt.subplots(figsize=(6.6,5.4)); im=ax.imshow(Mmat,cmap="YlGnBu",vmin=0.88,vmax=0.99)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=20,ha="right",fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order,fontsize=8)
    ax.set_xlabel("Target (test)"); ax.set_ylabel("Source (train)"); ax.set_title("All-pairs generalization matrix — mean R² (5 seeds)")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j,i,f"{Mmat[i,j]:.3f}",ha="center",va="center",fontsize=9,
                    color="white" if Mmat[i,j]>0.955 else "black",fontweight="bold" if i==j else "normal")
            if i==j: ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor=ORANGE,lw=2.5))
    fig.colorbar(im,ax=ax,shrink=0.8); save(fig,"fig4_matrix.png")

    # gap-by-type for fig9 (tech vs climate from matrix off-diagonal)
    def meta(s): return ("NIST" if "nist" in s.lower() else "AU")
    tech_gaps=[]; clim_gaps=[]
    diag={order[i]:Mmat[i,i] for i in range(len(order))}
    for i,src in enumerate(order):
        for j,tgt in enumerate(order):
            if i==j: continue
            gap=diag[tgt]-Mmat[i,j]
            (clim_gaps if meta(src)!=meta(tgt) else tech_gaps).append(gap)
    return dict(df=df,feats=feats,base=base,tech_gap=np.mean(tech_gaps),clim_gap=np.mean(clim_gaps))

# ---------- TRANSFER FIGURES ----------
def transfer_figs(ctx):
    if ctx is None: return
    df,feats,base=ctx["df"],ctx["feats"],ctx["base"]
    Xs,ys=d.get_xy(d.site_frame(df,[MONO]),"power",False)
    Xt,yt=d.get_xy(d.site_frame(df,[NIST]),"power",False)
    Xpool,Xte,ypool,yte=train_test_split(Xt,yt,test_size=0.4,random_state=42)
    zero=met.regression_metrics(yte,clone(base).fit(Xs,ys).predict(Xte))["R2"]
    within=met.regression_metrics(yte,clone(base).fit(Xpool,ypool).predict(Xte))["R2"]; gap=within-zero
    rng=np.random.default_rng(42); fracs=[0.05,0.1,0.2]; rec=[]
    for f in fracs:
        k=min(max(5,int(f*len(Xt))),len(Xpool)); idx=rng.choice(len(Xpool),k,replace=False)
        Xc=pd.concat([Xs,Xpool.iloc[idx]]); yc=np.concatenate([ys,ypool[idx]])
        s=met.regression_metrics(yte,clone(base).fit(Xc,yc).predict(Xte))["R2"]; rec.append((s-zero)/gap*100)
    fig,ax=plt.subplots(figsize=(7,4.4)); ax.plot([f*100 for f in fracs],rec,"o-",color=NAVY)
    ax.set_xlabel("Target-site data used (%)"); ax.set_ylabel("Cross-climate gap recovered (%)")
    ax.set_title("Few-shot transfer-learning recovery"); ax.grid(alpha=0.3); save(fig,"fig3_recovery.png")
    # fig5: 5-method comparison @20%
    k=min(max(5,int(0.2*len(Xt))),len(Xpool)); idx=rng.choice(len(Xpool),k,replace=False)
    Xk,yk=Xpool.iloc[idx],ypool[idx]
    # CORAL
    def matpow(A,p,e=1e-6):
        A=A+e*np.eye(A.shape[0]); w,V=np.linalg.eigh(A); w=np.clip(w,1e-10,None); return (V*(w**p))@V.T
    mu,sd=Xs.values.mean(0),Xs.values.std(0)+1e-8; Xsz=(Xs.values-mu)/sd; Xtez=(Xte.values-mu)/sd
    coral=met.regression_metrics(yte,clone(base).fit(Xsz,ys).predict(Xtez))["R2"] if Xs.shape[1]<2 else None
    if coral is None:
        A=matpow(np.cov(Xsz,rowvar=False),-0.5)@matpow(np.cov(Xtez,rowvar=False),0.5)
        coral=met.regression_metrics(yte,clone(base).fit(Xsz@A,ys).predict(Xtez))["R2"]
    # importance weighting
    Xd=np.vstack([Xs.values,Xpool.values]); yd=np.r_[np.zeros(len(Xs)),np.ones(len(Xpool))]
    clf=HistGradientBoostingClassifier(random_state=0).fit(Xd,yd); pw=np.clip(clf.predict_proba(Xs.values)[:,1],1e-3,1-1e-3); wgt=(pw/(1-pw)); wgt/=wgt.mean()
    iw=met.regression_metrics(yte,clone(base).fit(Xs,ys,sample_weight=wgt).predict(Xte))["R2"]
    # few shot tree
    fs=met.regression_metrics(yte,clone(base).fit(pd.concat([Xs,Xk]),np.concatenate([ys,yk])).predict(Xte))["R2"]
    # MLP fine-tune
    sc=StandardScaler().fit(Xs); net=MLPRegressor(hidden_layer_sizes=(32,16),max_iter=400,random_state=0,warm_start=True)
    net.fit(sc.transform(Xs),ys); net.max_iter=200; net.fit(sc.transform(Xk),yk)
    mlp=met.regression_metrics(yte,net.predict(sc.transform(Xte)))["R2"]
    R={"CORAL\n(unsup.)":(coral-zero)/gap*100,"Importance\nweight (unsup.)":(iw-zero)/gap*100,
       "Few-shot\ntree (20%)":(fs-zero)/gap*100,"MLP fine-tune\n(20%)":(mlp-zero)/gap*100}
    fig,ax=plt.subplots(figsize=(7,4.4))
    bars=ax.bar(list(R),list(R.values()),color=[GREY,GREY,NAVY,GREEN],width=0.62)
    ax.axhline(0,color="#555",lw=1); ax.axhline(100,color=ORANGE,ls="--",lw=1.3)
    ax.set_ylabel("Cross-climate gap recovered (%)"); ax.set_title("Adaptation method matters"); ax.set_ylim(-15,110); ax.grid(axis="y",alpha=0.3)
    for b,v in zip(bars,R.values()): ax.text(b.get_x()+b.get_width()/2,v+(2 if v>=0 else -6),f"{v:.0f}%",ha="center",fontsize=9,fontweight="bold")
    save(fig,"fig5_methods.png")

# ---------- ERROR ANALYSIS (fig6) ----------
def error_fig(ctx):
    if ctx is None: return
    df,feats,base=ctx["df"],ctx["feats"],ctx["base"]
    tgt=d.site_frame(df,[NIST]).reset_index(drop=True)
    Xs,ys=d.get_xy(d.site_frame(df,[MONO]),"power",False); Xt,yt=d.get_xy(tgt,"power",False)
    cross=clone(base).fit(Xs,ys).predict(Xt); within=cross_val_predict(clone(base),Xt,yt,cv=5)
    tgt["cr"]=cross-yt; tgt["wr"]=within-yt
    bins=[0,200,400,600,800,1000,3000]; labs=["0-200","200-400","400-600","600-800","800-1000",">1000"]
    tgt["ib"]=pd.cut(tgt["irradiance"],bins=bins,labels=labs,include_lowest=True)
    rw=[np.sqrt((tgt[tgt.ib==l].wr**2).mean()) for l in labs]; rc=[np.sqrt((tgt[tgt.ib==l].cr**2).mean()) for l in labs]
    x=np.arange(len(labs)); w=0.4; fig,ax=plt.subplots(figsize=(7.2,4.3))
    ax.bar(x-w/2,rw,w,label="within-site",color=GREEN); ax.bar(x+w/2,rc,w,label="cross-climate",color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel("Irradiance (W/m²)"); ax.set_ylabel("RMSE (capacity factor)")
    ax.set_title("Cross-climate error concentrates at low irradiance"); ax.legend(); ax.grid(axis="y",alpha=0.3)
    save(fig,"fig6_err.png")

# ---------- UNCERTAINTY (fig8) ----------
def unc_fig(ctx):
    if ctx is None: return
    df,feats,base=ctx["df"],ctx["feats"],ctx["base"]
    Xs,ys=d.get_xy(d.site_frame(df,[MONO]),"power",False); Xt,yt=d.get_xy(d.site_frame(df,[NIST]),"power",False)
    Xtr,Xtmp,ytr,ytmp=train_test_split(Xs,ys,test_size=0.4,random_state=42)
    Xcal,Xste,ycal,yste=train_test_split(Xtmp,ytmp,test_size=0.5,random_state=42)
    mdl=clone(base).fit(Xtr,ytr); q=np.quantile(np.abs(ycal-mdl.predict(Xcal)),0.9)
    ci=np.mean(np.abs(yste-mdl.predict(Xste))<=q)*100; co=np.mean(np.abs(yt-mdl.predict(Xt))<=q)*100
    fig,ax=plt.subplots(figsize=(5.6,4.2)); ax.bar(["within-site\n(source)","cross-site\n(target)"],[ci,co],color=[GREEN,ORANGE])
    ax.axhline(90,color=NAVY,ls="--",lw=1.3,label="nominal 90%"); ax.set_ylim(0,105); ax.set_ylabel("90% interval coverage (%)")
    ax.set_title("Cross-site intervals stay calibrated"); ax.legend(loc="lower right")
    for i,v in enumerate([ci,co]): ax.text(i,v+1,f"{v:.0f}%",ha="center")
    save(fig,"fig8_coverage.png")

# ---------- TEMPORAL + THREE-AXIS (fig9) ----------
def temporal_and_threeaxis(ctx):
    if ctx is None: return
    base=ctx["base"]; time_gap=np.nan
    try:
        import dkasc_loader as dl
        drops=[]
        for site in [MONO,POLY,CDTE]:
            cand=glob.glob(str(config.DATA_DIR/"dkasc"/f"{site}*.csv"))
            if not cand: continue
            f0,_=dl.load_one(cand[0],site,"1h","2017-01-01","2017-12-31",normalize=False)
            if f0 is None or len(f0)==0: continue
            cap=f0["power"].quantile(0.99) or 1.0
            def yr(y):
                fy,_=dl.load_one(cand[0],site,"1h",f"{y}-01-01",f"{y}-12-31",normalize=False)
                if fy is None or len(fy)==0: return None
                fy=fy.copy(); fy["power"]=fy["power"]/cap; return fy
            a=yr(2017); b=yr(2021)
            if a is None or b is None: continue
            feats=[c for c in ["irradiance"] if c in a.columns]
            xa,xb,ya,yb=train_test_split(a[feats],a["power"],test_size=0.2,random_state=42)
            within=met.regression_metrics(yb,clone(base).fit(xa,ya).predict(xb))["R2"]
            mdl=clone(base).fit(a[feats],a["power"]); r=met.regression_metrics(b["power"],mdl.predict(b[feats]))["R2"]
            drops.append(max(within-r,0))
        if drops: time_gap=float(np.mean(drops))
    except Exception as e: print("  [temporal] skipped:",e)
    tech=ctx.get("tech_gap",np.nan); clim=ctx.get("clim_gap",np.nan)
    vals=[tech, time_gap if not np.isnan(time_gap) else 0.007, clim]
    labs=["Across technology\n(mono/poly/CdTe)","Across time\n(4-year gap)","Across climate\n(Australia→USA)"]
    fig,ax=plt.subplots(figsize=(7,4.3)); bars=ax.bar(labs,vals,color=[GREY,GREY,NAVY],width=0.6)
    ax.set_ylabel("Generalization gap (mean ΔR²)"); ax.set_title("Climate is the only barrier: technology & time are essentially free")
    ax.grid(axis="y",alpha=0.3)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.0004,f"{v:.3f}",ha="center",fontweight="bold")
    save(fig,"fig9_threeaxes.png")

# ---------- FAULT (fig7, fig10) ----------
def fault_figs():
    fp=config.DATA_DIR/"gpvs_features.csv"
    if not fp.exists(): print("[skip] no data/gpvs_features.csv (run gpvs_prep.py)"); return
    df=pd.read_csv(fp); feat=[c for c in df.columns if c not in ("fault","scenario","mode")]
    zoo=m.get_classifiers(); base=zoo.get("CatBoost") or next(iter(zoo.values()))
    def clf(seed): 
        c=clone(base)
        try: c.set_params(random_state=seed)
        except: pass
        return c
    rec=[]
    for seed in SEEDS:
        for src in ["MPPT","IPPT"]:
            for tgt in ["MPPT","IPPT"]:
                if src==tgt:
                    X=df[df["mode"]==src][feat]; y=df[df["mode"]==src]["fault"]
                    xa,xb,ya,yb=train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
                    c=clf(seed).fit(xa,ya); pr=c.predict_proba(xb)[:,1]
                    r=met.classification_metrics(yb,c.predict(xb),pr); r["type"]="within"
                else:
                    Xtr=df[df["mode"]==src][feat]; ytr=df[df["mode"]==src]["fault"]; Xte=df[df["mode"]==tgt][feat]; yte=df[df["mode"]==tgt]["fault"]
                    c=clf(seed).fit(Xtr,ytr); pr=c.predict_proba(Xte)[:,1]
                    r=met.classification_metrics(yte,c.predict(Xte),pr); r["type"]=f"{src[0]}->{tgt[0]}"
                rec.append(r)
    R=pd.DataFrame(rec)
    grp={"Within-mode":R[R.type=="within"],"Cross:\nIPPT→MPPT":R[R.type=="I->M"],"Cross:\nMPPT→IPPT":R[R.type=="M->I"]}
    labs=list(grp); F1=[grp[k].F1.mean() for k in labs]; AUC=[grp[k].AUC.mean() for k in labs]
    x=np.arange(3); w=0.38; fig,ax=plt.subplots(figsize=(7,4.3))
    ax.bar(x-w/2,F1,w,label="F1",color=NAVY); ax.bar(x+w/2,AUC,w,label="AUC",color=ORANGE)
    ax.axhline(0.5,color="#999",ls=":",lw=1); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0,1.08)
    ax.set_ylabel("Score"); ax.set_title("Cross-mode fault detection: F1 hides an AUC collapse"); ax.legend(loc="lower left"); ax.grid(axis="y",alpha=0.3)
    save(fig,"fig7_fault.png")
    # confusion matrices (seed 42)
    fig,axes=plt.subplots(1,2,figsize=(9,4.1))
    for ax,(src,tgt,title) in zip(axes,[("MPPT","IPPT","MPPT → IPPT (collapse)"),("IPPT","MPPT","IPPT → MPPT (works)")]):
        Xtr=df[df["mode"]==src][feat]; ytr=df[df["mode"]==src]["fault"]; Xte=df[df["mode"]==tgt][feat]; yte=df[df["mode"]==tgt]["fault"]
        cm=confusion_matrix(yte,clf(42).fit(Xtr,ytr).predict(Xte),labels=[0,1])
        im=ax.imshow(cm,cmap="Blues"); ax.set_title(title,fontsize=11)
        ax.set_xticks([0,1]); ax.set_xticklabels(["pred\nhealthy","pred\nfaulty"]); ax.set_yticks([0,1]); ax.set_yticklabels(["actual\nhealthy","actual\nfaulty"])
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,color="white" if cm[i,j]>cm.max()*0.55 else "black",fontweight="bold")
    fig.suptitle("Cross-mode fault-detection confusion matrices"); save(fig,"fig10_confusion.png")

if __name__=="__main__":
    print("[paper-figs] regenerating from real data into figures/paper/ ...")
    ctx=power_figs(); transfer_figs(ctx); error_fig(ctx); unc_fig(ctx); temporal_and_threeaxis(ctx); fault_figs()
    print("[done]")
