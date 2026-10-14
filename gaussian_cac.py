"""Correlated Gaussian log-parameters and noisy scores: reproducible experiment.

python3 gaussian_cac.py --power 14 --replicates 4 --order 20

Posterior means use Gaussian exponential tilting plus 2D Gauss-Hermite
quadrature. Outer expectations use independently scrambled Sobol replicates.
Reported numerical standard errors are integration diagnostics, not data CIs.
"""
import argparse
import json
import os
from itertools import combinations, product
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '.mplconfig')))
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtri
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MU = np.log([.08, .8, 100.])
SD = np.array([.55, .65, .60])
CORR = np.array([[1., -.45, -.30], [-.45, 1., .55], [-.30, .55, 1.]])
SIGMA = CORR * np.outer(SD, SD)
NOISE_SD = np.array([1., .75, .4])
D = np.diag(NOISE_SD**2)
HORIZON, MARGIN, COST, PRICE = 12, .1, 3., 25.
SUBSETS = [j for size in range(4) for j in combinations(range(3), size)]
LABELS = ['Без скоров', 'Отток', 'Активность', 'Оборот',
          'Отток + активность', 'Отток + оборот', 'Активность + оборот', 'Все три']

def key(j):
    return ''.join(str(i) for i in j) or 'none'

def duration(h, n=HORIZON):
    lam = np.exp(h)
    return -np.expm1(-n*lam) / -np.expm1(-lam)

def activity(a):
    return -np.expm1(-np.exp(a))

def cash(z, n=HORIZON):
    return duration(z[...,0], n)*(MARGIN*np.exp(z[...,2])*activity(z[...,1])-COST)

def posterior(w, subset, noise=D):
    if not subset:
        return np.broadcast_to(MU, (len(w),3)).copy(), SIGMA.copy()
    j = list(subset)
    cross = SIGMA[:,j]
    gain = np.linalg.solve(SIGMA[np.ix_(j,j)]+noise[np.ix_(j,j)],cross.T).T
    means = MU + (w[:,j]-MU[j])@gain.T
    cov = SIGMA-gain@cross.T
    return means, (cov+cov.T)/2

def normal_grid(order, dim):
    x, weights = hermgauss(order)
    ix = np.array(list(product(range(order),repeat=dim)))
    return np.sqrt(2)*x[ix], np.prod(weights[ix]/np.sqrt(np.pi),axis=1)

def posterior_cash(mean, cov, order=20, second=False):
    """Exact integral formula; numerical Gaussian quadrature evaluation."""
    nodes, weights = normal_grid(order,2)
    offsets = nodes@np.linalg.cholesky(cov[:2,:2]).T
    xn, wn = normal_grid(order,1)
    out = np.empty(len(mean))
    for begin in range(0,len(mean),512):
        sl = slice(begin,begin+512)
        b = mean[sl]
        u = b[:,None,:2]+offsets[None,:,:]
        hs = b[:,0,None]+np.sqrt(cov[0,0])*xn[:,0][None,:]
        if not second:
            shifted = u+cov[:2,2]
            revenue = (duration(shifted[:,:,0])*activity(shifted[:,:,1]))@weights
            costs = duration(hs)@wn
            out[sl] = MARGIN*np.exp(b[:,2]+cov[2,2]/2)*revenue-COST*costs
        else:
            shifted1, shifted2 = u+cov[:2,2], u+2*cov[:2,2]
            first = (duration(shifted2[:,:,0])**2*activity(shifted2[:,:,1])**2)@weights
            middle = (duration(shifted1[:,:,0])**2*activity(shifted1[:,:,1]))@weights
            last = duration(hs)**2@wn
            out[sl] = (MARGIN**2*np.exp(2*b[:,2]+2*cov[2,2])*first
                       -2*MARGIN*COST*np.exp(b[:,2]+cov[2,2]/2)*middle+COST**2*last)
    return out

def derivatives(z):
    lam, nu, v = np.exp(z)
    t = np.arange(HORIZON)
    q = t*lam
    e = np.exp(-q)
    l, l1, l2 = e.sum(), (-q*e).sum(), ((q*q-q)*e).sum()
    a, a1, a2 = -np.expm1(-nu), nu*np.exp(-nu), nu*np.exp(-nu)*(1-nu)
    revenue = MARGIN*v
    b = revenue*a-COST
    grad = np.array([l1*b,l*revenue*a1,l*revenue*a])
    hess = np.array([[l2*b,l1*revenue*a1,l1*revenue*a],
                     [l1*revenue*a1,l*revenue*a2,l*revenue*a1],
                     [l1*revenue*a,l*revenue*a1,l*revenue*a]])
    return grad,hess

def metrics(pred, truth, oracle):
    return {'mse':float(np.mean((pred-truth)**2)),
            'bias':float(np.mean(pred-truth)),
            'regret':float(np.mean(oracle-(truth-PRICE)*(pred>=PRICE)))}

def validate(order):
    assert np.linalg.eigvalsh(SIGMA).min()>0
    z = MU + np.array([.1,-.1,.2])
    grad,hess=derivatives(z)
    eps=1e-4
    basis=np.eye(3)*eps
    for i in range(3):
        finite=(cash(z+basis[i])-cash(z-basis[i]))/(2*eps)
        assert abs(finite-grad[i])<2e-6
        for j in range(3):
            finite2=(cash(z+basis[i]+basis[j])-cash(z+basis[i]-basis[j])
                     -cash(z-basis[i]+basis[j])+cash(z-basis[i]-basis[j]))/(4*eps**2)
            assert abs(finite2-hess[i,j])<1e-5
    # Check the reduction from 3D integration against direct quadrature.
    means,cov=posterior(np.array([MU, z]),(0,1,2))
    nodes,weights=normal_grid(28,3)
    points=means[:,None,:]+nodes@np.linalg.cholesky(cov).T
    direct=cash(points)@weights
    tilted=posterior_cash(means,cov,28)
    assert np.max(np.abs(direct-tilted))<1e-7
    direct2=cash(points)**2@weights
    tilted2=posterior_cash(means,cov,28,True)
    assert np.max(np.abs(direct2-tilted2))<1e-5
    # No-noise posterior collapses to the true parameters.
    exactmean,exactcov=posterior(np.array([z]),(0,1,2),np.zeros((3,3)))
    assert np.max(np.abs(exactmean[0]-z))<1e-12
    assert np.max(np.abs(exactcov))<1e-12
    # Duration agrees with its finite survival sum, including n=1.
    assert abs(duration(z[0])-np.exp(-np.arange(HORIZON)*np.exp(z[0])).sum())<1e-12
    assert abs(duration(z[0],1)-1)<1e-12
    return {'direct_3d_vs_tilted_max_abs':float(np.max(np.abs(direct-tilted))),
            'second_moment_max_abs':float(np.max(np.abs(direct2-tilted2)))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--power',type=int,default=14)
    parser.add_argument('--replicates',type=int,default=4)
    parser.add_argument('--order',type=int,default=20)
    args=parser.parse_args()
    verification=validate(args.order)
    records=[]
    for replicate in range(args.replicates):
        uniform=qmc.Sobol(6,scramble=True,seed=20261013+replicate).random_base2(args.power)
        normal=ndtri(uniform)
        z=MU+normal[:,:3]@np.linalg.cholesky(SIGMA).T
        w=z+normal[:,3:]*NOISE_SD
        truth=cash(z)
        oracle=np.maximum(truth-PRICE,0)
        row={}
        bayes={}
        for subset in SUBSETS:
            means,cov=posterior(w,subset)
            pred=posterior_cash(means,cov,args.order)
            bayes[key(subset)]=pred
            raw=np.broadcast_to(MU,w.shape).copy()
            raw[:,list(subset)]=w[:,list(subset)]
            row[key(subset)]={
                'bayes':metrics(pred,truth,oracle),
                'posterior_mean_plugin':metrics(cash(means),truth,oracle),
                'raw_plugin':metrics(cash(raw),truth,oracle)}
        # The L2 projection identity: improvement equals squared forecast update.
        identity=[]
        for subset in SUBSETS:
            for j in range(3):
                if j in subset:
                    continue
                larger=tuple(sorted(subset+(j,)))
                delta=bayes[key(larger)]-bayes[key(subset)]
                identity.append({'from':key(subset),'to':key(larger),
                    'squared_update':float(np.mean(delta**2)),
                    'mse_difference':row[key(subset)]['bayes']['mse']-row[key(larger)]['bayes']['mse']})
        if replicate==0:
            # Quadrature refinement on points spread across the sample.
            selected=np.linspace(0,len(w)-1,min(1024,len(w)),dtype=int)
            maxerr=0.
            for subset in SUBSETS:
                means,cov=posterior(w[selected],subset)
                fine=posterior_cash(means,cov,args.order+12)
                maxerr=max(maxerr,float(np.max(np.abs(fine-bayes[key(subset)][selected]))))
            verification['quadrature_refinement_max_abs']=maxerr
            # Exact Gaussian covariance update versus one-step formula.
            _,vold=posterior(w[:1],(2,))
            _,vnew=posterior(w[:1],(0,2))
            update=np.outer(vold[:,0],vold[:,0])/(vold[0,0]+D[0,0])
            assert np.max(np.abs(vold-vnew-update))<1e-12
        records.append({'metrics':row,'projection_identities':identity,
                        'true_mean':float(truth.mean()),'oracle_profit':float(oracle.mean())})
        print('Completed replicate',replicate+1,flush=True)
    summary={}
    for subset in SUBSETS:
        k=key(subset)
        summary[k]={}
        for method in ('bayes','posterior_mean_plugin','raw_plugin'):
            summary[k][method]={}
            for metric in ('mse','bias','regret'):
                values=np.array([r['metrics'][k][method][metric] for r in records])
                summary[k][method][metric]={'mean':float(values.mean()),
                    'integration_se':float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None}
    # Prior moments evaluated independently by quadrature.
    prior_mean=float(posterior_cash(MU[None,:],SIGMA,40)[0])
    prior_second=float(posterior_cash(MU[None,:],SIGMA,40,True)[0])
    result={'parameters':{'mu':MU.tolist(),'latent_sd':SD.tolist(),'latent_corr':CORR.tolist(),
                         'noise_sd':NOISE_SD.tolist(),'horizon':HORIZON,'margin':MARGIN,
                         'opex':COST,'acquisition_price':PRICE},
            'integration':vars(args),'verification':verification,'summary':summary,
            'prior_cash_mean':prior_mean,'prior_cash_variance':prior_second-prior_mean**2,
            'replicates':records}
    (ROOT/'gaussian_results.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
    selected=[(2,),(1,2),(0,1,2)]
    x=np.arange(3)
    for offset,method,label,color in [(-.18,'raw_plugin','Прямая подстановка','#d27c38'),
                                       (.18,'bayes','Условное ожидание CAC','#278477')]:
        for ax,metric in zip(axes,['mse','regret']):
            means=[summary[key(j)][method][metric]['mean'] for j in selected]
            errors=[summary[key(j)][method][metric]['integration_se'] or 0 for j in selected]
            ax.bar(x+offset,means,width=.34,label=label,color=color,yerr=errors,capsize=3)
            ax.set_xticks(x,['Оборот','Активность\n+ оборот','Все три'])
            ax.grid(axis='y',alpha=.2)
            ax.set_axisbelow(True)
    axes[0].set_title('Ошибка оценки истинного CAC')
    axes[0].set_ylabel('Средний квадрат ошибки')
    axes[1].set_title('Потеря прибыли при цене привлечения 25')
    axes[1].set_ylabel('Потеря прибыли на кандидата')
    axes[0].legend(frameon=False,fontsize=9)
    fig.savefig(ROOT/'gaussian_cac_comparison.png',dpi=180)
    print(json.dumps({'verification':verification,'summary':summary},indent=2))

if __name__=='__main__':
    main()
