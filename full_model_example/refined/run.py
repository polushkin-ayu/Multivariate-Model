"""Full cash model, known common calibration, new-score pilot uncertainty."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mplconfig')))
import sys, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from numpy.polynomial.legendre import leggauss
from verify_structural_weak_score import model as initial_model
from cac_calibration import GaussianCAC
def model():
    m=initial_model();d=m.noise.copy();d[0,0]=4.;d[1,1]*=4;d[2,2]*=4
    return GaussianCAC(m.mu,m.sigma,d,m.cash)
from structural_weak_score import WeakBoundaryEngine, predict
from cac_decision import BoundaryEngine

OUT=Path(__file__).resolve().parent
TAU=.25; PRICE=75.; FLOW=100000; REPS=200000; SEED=2026101347
GATE=(-3.200874778243636,2.7545876756152166,3.)

def plain(o):
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,dict): return {k:plain(v) for k,v in o.items()}
    if isinstance(o,(tuple,list)): return [plain(v) for v in o]
    return o

def baseline(m,outer=40,inner=28,vertical=72):
    e=BoundaryEngine(m,(1,2),PRICE,outer,inner)
    r=e.root
    nodes,weights=leggauss(vertical)
    top=np.maximum(r,e.conditional_mean+12*e.conditional_sd)
    v=r[:,None]+(top-r)[:,None]*(nodes+1)/2
    w=np.repeat(e.w,vertical,axis=0);w[:,e.vcol]=v.ravel()
    p=(m.predict(w,(1,2),inner).reshape(v.shape)-PRICE)
    density=norm.pdf((v-e.conditional_mean[:,None])/e.conditional_sd)/e.conditional_sd
    profit=e.weights@(np.sum(p*density*weights,axis=1)*(top-r)/2)
    accept=e.weights@norm.sf((r-e.conditional_mean)/e.conditional_sd)
    return {'profit':float(profit),'acquisition_probability':float(accept),'cac_spend':float(PRICE*accept),'cash_before_cac':float(profit+PRICE*accept),'roi':float(profit/(PRICE*accept))}

def effects(e,m,t,base,vertical_order=16):
    root=e.root(m,t,1.)
    nodes,weights=leggauss(vertical_order)
    v=root[:,None]+(base-root)[:,None]*(nodes+1)/2
    w=np.column_stack([np.repeat(e.nuisance[:,0],vertical_order),v.ravel()])
    x=np.repeat(e.nuisance[:,1],vertical_order)
    payoff=predict(m,w,x,TAU,e.order).reshape(v.shape)-PRICE
    den=norm.pdf((v-e.vmean[:,None])/e.vsd)/e.vsd
    signed=(base-root)[:,None]/2*weights*den*e.weights[:,None]
    gain=float(np.sum(signed*payoff))
    added=signed>0
    beneficial=np.maximum(signed*payoff,0).sum()
    harmful=-np.minimum(signed*payoff,0).sum()
    da=float(e.weights@(norm.cdf((base-e.vmean)/e.vsd)-norm.cdf((root-e.vmean)/e.vsd)))
    return gain,da,float(beneficial),float(harmful)

def calibration_draws(m,n,reps,rng):
    # Sample covariance of the Gaussian pair (mean H reference, X).
    # Bartlett: S_HX=c*A/df + sqrt(vh*vx-c*c)*sqrt(A)*Z/df.
    df=n-1; vh=m.sigma[0,0]+m.noise[0,0]/2; vx=1+TAU**2*m.sigma[0,0]; c=TAU*m.sigma[0,0]
    a=rng.chisquare(df,reps); z=rng.normal(size=reps)
    raw=(c*a+np.sqrt(vh*vx-c*c)*np.sqrt(a)*z)/(df*m.sigma[0,0])
    return raw,np.clip(raw,-1,1)

def main():
    m=model();start=time.time()
    grid=np.linspace(-1,1,161)
    e=WeakBoundaryEngine(m,TAU,PRICE,18,24,gate=GATE)
    br=e.root(m,0,0)
    cache=OUT/'curve_checkpoint.json'
    curve=json.loads(cache.read_text()) if cache.exists() else []
    for j,t in enumerate(grid):
        if j<len(curve):
            assert abs(curve[j]['fitted_tau']-t)<1e-12
            continue
        vals=effects(e,m,float(t),br)
        curve.append(dict(fitted_tau=float(t),gain=vals[0],delta_accept=vals[1],beneficial=vals[2],harmful=vals[3]))
        cache.write_text(json.dumps(curve))
        if j%20==0: print(f'curve {j+1}/{len(grid)}, {time.time()-start:.1f}s',flush=True)
    interpolators={k:PchipInterpolator(grid,[q[k] for q in curve]) for k in ['gain','delta_accept','beneficial','harmful']}
    # Higher-order fixed-knot checks, independent of observed effect signs.
    high=WeakBoundaryEngine(m,TAU,PRICE,28,36,gate=GATE); highbr=high.root(m,0,0)
    checks=[]
    for t in [-1,-.5,-.00625,0,.00625,.25,.50625,1]:
        val=effects(high,m,t,highbr)
        checks.append(dict(tau=t,exact_high=list(val),interpolated=[float(interpolators[k](t)) for k in interpolators],gain_error=float(interpolators['gain'](t)-val[0])))
    base=baseline(m); basehigh=baseline(m,64,40,96)
    oracle=effects(high,m,TAU,highbr)[0]
    rng=np.random.default_rng(SEED)
    ns=np.unique(np.r_[np.rint(np.geomspace(32,4096,33)).astype(int),64,128,256,512,1024])
    rows=[]; primary=None
    for n in ns:
        raw,th=calibration_draws(m,int(n),REPS,rng)
        g=interpolators['gain'](th); da=interpolators['delta_accept'](th)
        bene=interpolators['beneficial'](th); harm=interpolators['harmful'](th)
        row=dict(N=int(n),replications=REPS,mean_gain=float(g.mean()),mc_se=float(g.std(ddof=1)/np.sqrt(REPS)),q05=float(np.quantile(g,.05)),q50=float(np.median(g)),q95=float(np.quantile(g,.95)),fraction_worse=float(np.mean(g<0)),mean_delta_accept=float(da.mean()),beneficial=float(bene.mean()),harmful=float(harm.mean()),clipped_fraction=float(np.mean(abs(raw)>1)),tauhat_mean=float(th.mean()),tauhat_sd=float(th.std()))
        row['profit']=basehigh['profit']+row['mean_gain'];row['cac_spend']=basehigh['cac_spend']+PRICE*row['mean_delta_accept'];row['roi_ratio_of_expectations']=row['profit']/row['cac_spend']
        rows.append(row)
        if n==64:
            primary=row
            counts,edges=np.histogram(th,bins=np.linspace(-1,1,51))
            pilot=dict(counts=counts.tolist(),edges=edges.tolist(),gain_quantiles=np.quantile(g,np.linspace(0,1,201)).tolist())
    # Direct reference sample cross-check: sampling law, not same random stream.
    reps=20000;n=64
    h=rng.normal(0,np.sqrt(m.sigma[0,0]),(reps,n));ref=h+rng.normal(0,np.sqrt(m.noise[0,0]/2),(reps,n));xx=TAU*h+rng.normal(size=(reps,n))
    direct=((ref-ref.mean(1,keepdims=True))*(xx-xx.mean(1,keepdims=True))).sum(1)/(n-1)/m.sigma[0,0]
    vh=m.sigma[0,0]+m.noise[0,0]/2;vx=1+TAU**2*m.sigma[0,0];c=TAU*m.sigma[0,0]
    sd=np.sqrt((vh*vx+c*c)/(n-1)/m.sigma[0,0]**2)
    result=dict(design=dict(tau=TAU,flow=FLOW,price=PRICE,gate=GATE,seed=SEED,known_common_calibration=True,primary_N=64),score_quality=dict(corr_X_H=TAU*np.sqrt(m.sigma[0,0])/np.sqrt(vx),R2_H_from_X=TAU**2*m.sigma[0,0]/vx,unclipped_estimator_sd_N64=sd),baseline=basehigh,baseline_order_check=base,oracle_gain=oracle,curve=curve,checks=checks,calibration_law_check=dict(direct_mean=float(direct.mean()),direct_sd=float(direct.std()),theory_mean=TAU,theory_sd=sd,direct_reps=reps),primary=primary,pilot_distribution=pilot,by_N=rows,elapsed_seconds=time.time()-start)
    (OUT/'results.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    print(json.dumps(plain({k:result[k] for k in ['score_quality','baseline','oracle_gain','primary','calibration_law_check','elapsed_seconds']}),indent=2),flush=True)

if __name__=='__main__':main()
