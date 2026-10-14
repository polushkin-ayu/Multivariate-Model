"""Reproduce the structural weak-score checks (all data are synthetic)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from cac_calibration import GaussianCAC, CashSpec, moment_asymptotic_covariance
from cac_decision import pack, unpack, regularity_certificate
from structural_weak_score import (predict, slope, fit, soft_weight,
                                   boundary_constants, WeakBoundaryEngine)

ROOT = Path(__file__).resolve().parent


def model():
    sd = np.array([.55,.65,.60])
    corr = np.array([[1.,-.45,-.30],[-.45,1.,.55],[-.30,.55,1.]])
    return GaussianCAC(np.log([.08,.8,100.]),corr*np.outer(sd,sd),
                       np.diag([1.,.75**2,.4**2]),CashSpec())


def sample(m, n, tau, rng, common_error=0.):
    z = rng.multivariate_normal(m.mu,m.sigma,n)
    errors = rng.normal(size=(n,2,3))*np.sqrt(np.diag(m.noise))
    refs = z[:,None,:]+errors
    # Optional violation: eta correlated with first reference H error.
    eta = (np.sqrt(1-common_error**2)*rng.normal(size=n)
           +common_error*errors[:,0,0]/np.sqrt(m.noise[0,0]))
    x = tau*(z[:,0]-m.mu[0])+eta
    return refs,x


def limiting_risk(t, threshold):
    def f(e):
        y = t+e
        b = np.sign(y)*max(abs(y)-threshold,0.)
        return (b-t)**2*norm.pdf(e)
    cuts = [-np.inf,-threshold-t,threshold-t,np.inf]
    return sum(quad(f,a,b,epsabs=1e-10)[0] for a,b in zip(cuts[:-1],cuts[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal-reps',type=int,default=1500)
    parser.add_argument('--profit-reps',type=int,default=32)
    args = parser.parse_args()
    rng = np.random.default_rng(2026101307)
    m = model()
    s = np.sqrt((m.sigma[0,0]+m.noise[0,0]/2)/m.sigma[0,0]**2)
    w = m.mu[[1,2]]+rng.normal(size=(20,2))*.4
    x = rng.normal(size=20)
    assert np.max(abs(predict(m,w,x,0)-m.predict(w,(1,2),24)))<1e-12
    eps = 1e-4
    fd = (predict(m,w,x,eps,32)-predict(m,w,x,-eps,32))/(2*eps)
    derivative_error = float(np.max(abs(fd-x*slope(m,w,32))))
    assert derivative_error<2e-6
    # Independent direct joint-normal conditioning check.
    tau = .3
    load = np.array([[0,1,0],[0,0,1],[tau,0,0.]])
    cov = load@m.sigma@load.T+np.diag([m.noise[1,1],m.noise[2,2],1.])
    gain = np.linalg.solve(cov,load@m.sigma).T
    means = np.array([m.mu[1],m.mu[2],0.])
    b = m.mu+(np.column_stack([w,x])-means)@gain.T
    q = m.sigma-gain@load@m.sigma
    from structural_weak_score import mean_cash
    joint_error = float(np.max(abs(predict(m,w,x,tau,28)-mean_cash(b,q,m.cash,28))))
    assert joint_error<1e-10
    const = boundary_constants(m,outer_order=20,inner_order=32)
    const2 = boundary_constants(m,outer_order=28,inner_order=40)
    C = const2['C']
    assert abs(C-const['C'])<1e-5
    V = moment_asymptotic_covariance(m.sigma,m.noise,2)
    B0 = float(np.trace(const2['base_matrix']@V))
    result = {'scope':'synthetic; reference-augmented calibration; asymptotic guarantee only',
              's':float(s),'C':C,'B0':B0,
              'regularity_certificate':regularity_certificate(m,(1,2)),
              'checks':{'zero_signal_exact':True,'slope_max_abs':derivative_error,
                        'joint_normal_max_abs':joint_error,
                        'C_quadrature_difference':abs(C-const['C'])},
              'calibration':[], 'local_paths':[], 'profit':[], 'limit':[]}
    for t in [0,1,2,3,4]:
        risk = limiting_risk(t,norm.ppf(.975))
        result['limit'].append({'h_over_s':t,'base_risk_normalized':t*t,
                                'full_risk_normalized':1.,'soft_risk_normalized':risk,
                                'soft_gain_over_base_normalized':t*t-risk})
    print('Posterior and derivative checks passed; C=',C,'s=',s,flush=True)

    # Deterministic local paths include a perturbed shared calibration.
    u = np.array([.2,-.1,.15,.03,.01,-.01,.02,.01,-.02,.1,.05,-.03])
    for t in [0,1,3]:
        h=t*s
        T=h+.7*s
        weight=.6
        leading=C*(2*h*weight*T-(weight*T)**2)
        for n in [1000,10000,100000]:
            fitted=unpack(pack(m)+u/np.sqrt(n),m.cash)
            engine=WeakBoundaryEngine(m,h/np.sqrt(n),outer_order=16,inner_order=28)
            scaled=n*engine.gain(fitted,T/np.sqrt(n),weight)
            result['local_paths'].append({'N':n,'h_over_s':t,'scaled_actual_gain':scaled,
                                          'leading_gain':leading,'error':scaled-leading})
    assert max(abs(r['error']) for r in result['local_paths'] if r['N']==100000)<.025
    print('Deterministic shared-calibration paths checked',flush=True)

    for n in [400,1600]:
        for t in [0,1,3]:
            h=t*s
            rows=[]; cover=[]; clipped=0
            for _ in range(args.cal_reps):
                refs,score=sample(m,n,h/np.sqrt(n),rng)
                fitted,estimate,shat,diagnostics=fit(refs,score,m.cash)
                rows.append(np.r_[np.sqrt(n)*(pack(fitted)-pack(m)),np.sqrt(n)*estimate-h])
                cover.append(abs(np.sqrt(n)*estimate-h)<=norm.ppf(.975)*shat)
                clipped+=diagnostics['tau_clipped']
            rows=np.array(rows)
            corr=np.corrcoef(rows,rowvar=False)
            result['calibration'].append({'N':n,'h_over_s':t,'replications':args.cal_reps,
                'standardized_bias':float(rows[:,-1].mean()/s),
                'variance_ratio':float(rows[:,-1].var(ddof=1)/s**2),
                'max_abs_nuisance_correlation':float(np.max(abs(corr[:-1,-1]))),
                'coverage':float(np.mean(cover)),'tau_clips':clipped})
    print('Joint calibration simulation finished',flush=True)

    for n in [2000,20000]:
        for t in [0,1,3]:
            h=t*s
            engine=WeakBoundaryEngine(m,h/np.sqrt(n),outer_order=8,inner_order=18)
            full=[]; soft=[]; checks=[]; weights=[]; repair=0
            for repeat in range(args.profit_reps):
                refs,score=sample(m,n,h/np.sqrt(n),rng)
                fitted,estimate,shat,diagnostics=fit(refs,score,m.cash)
                repair+=diagnostics['sigma_repair_frobenius']>1e-6
                lam=soft_weight(estimate,shat/np.sqrt(n))
                base_root=engine.root(fitted,0.,0.)
                full.append(n*engine.gain(fitted,estimate,1.,base_root))
                soft.append(n*engine.gain(fitted,estimate,lam,base_root))
                weights.append(lam)
                if repeat==0:
                    refined=WeakBoundaryEngine(m,h/np.sqrt(n),outer_order=16,inner_order=28)
                    fine=n*refined.gain(fitted,estimate,1.)
                    checks.append(abs(fine-full[-1]))
            full=np.array(full);soft=np.array(soft)
            def summary(values):
                return {'mean_scaled_gain':float(values.mean()),
                        'MC_SE_scaled_gain':float(values.std(ddof=1)/np.sqrt(len(values))),
                        'fraction_negative':float(np.mean(values < -1e-7))}
            row={'N':n,'h_over_s':t,'replications':args.profit_reps,
                 'full':summary(full),'soft':summary(soft),
                 'full_limit':C*s*s*(t*t-1),
                 'soft_limit':C*s*s*(t*t-limiting_risk(t,norm.ppf(.975))),
                 'soft_nonzero_fraction':float(np.mean(np.array(weights)>0)),
                 'one_replication_quadrature_check':max(checks),
                 'covariance_repairs':repair}
            result['profit'].append(row)
            print('Profit',n,t,row['full'],row['soft'],flush=True)
            (ROOT/'structural_weak_score_results.json').write_text(json.dumps(result,indent=2)+'\n')
    # Stress test of IDENTIFICATION, not a valid instance of the theorem.
    refs,score=sample(m,100000,0.,rng,common_error=.15)
    _,estimate,_,_=fit(refs,score,m.cash)
    result['common_error_stress']={'N':100000,'true_tau':0.,'correlation':.15,
        'estimated_tau':estimate,'probability_limit':.15*np.sqrt(m.noise[0,0])/(2*m.sigma[0,0])}
    (ROOT/'structural_weak_score_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved structural_weak_score_results.json',flush=True)


if __name__=='__main__':
    main()
