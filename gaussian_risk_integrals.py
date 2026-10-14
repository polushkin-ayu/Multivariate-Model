"""Deterministic checks of Bayes MSE using a conditional-replica identity.

If Z1,Z2 are independent given the same score W_J, then
E[E(G(Z)|W_J)^2] = E[G(Z1)G(Z2)].  Their cross covariance is
Sigma - V_J.  Gaussian tilting reduces the expectation to 4D quadrature.
"""
import json
import numpy as np
from scipy.special import ndtr
from gaussian_cac import (ROOT, MU, SIGMA, SUBSETS, MARGIN, COST, PRICE,
                          posterior, posterior_cash, normal_grid, duration,
                          activity, key)

def cross_moment(cross, order):
    joint=np.block([[SIGMA,cross],[cross.T,SIGMA]])
    u=[0,1,3,4]
    cov=joint[np.ix_(u,u)]
    mean=np.concatenate((MU[:2],MU[:2]))
    nodes,weights=normal_grid(order,4)
    base=mean+nodes@np.linalg.cholesky(cov).T
    def integral(tilt, both):
        x=base+joint[np.ix_(u,tilt)].sum(axis=1)
        val=duration(x[:,0])*activity(x[:,1])*duration(x[:,2])
        if both:
            val*=activity(x[:,3])
        return float(val@weights)
    revenue2=np.exp(2*MU[2]+SIGMA[2,2]+cross[2,2])*integral([2,5],True)
    revenue_cost=np.exp(MU[2]+SIGMA[2,2]/2)*integral([2],False)
    hh=joint[np.ix_([0,3],[0,3])]
    nodes2,weights2=normal_grid(order,2)
    h=MU[0]+nodes2@np.linalg.cholesky(hh).T
    cost2=float((duration(h[:,0])*duration(h[:,1]))@weights2)
    return MARGIN**2*revenue2-2*MARGIN*COST*revenue_cost+COST**2*cost2

def oracle_profit(order):
    """Integrate positive net profit, using conditional lognormal call formula."""
    nodes,weights=normal_grid(order,2)
    u=MU[:2]+nodes@np.linalg.cholesky(SIGMA[:2,:2]).T
    gain=np.linalg.solve(SIGMA[:2,:2],SIGMA[:2,2])
    vm=MU[2]+(u-MU[:2])@gain
    vs=np.sqrt(SIGMA[2,2]-SIGMA[2,:2]@gain)
    b=MARGIN*duration(u[:,0])*activity(u[:,1])
    a=COST*duration(u[:,0])+PRICE
    d2=(vm-np.log(a/b))/vs
    call=b*np.exp(vm+vs*vs/2)*ndtr(d2+vs)-a*ndtr(d2)
    return float(call@weights)

def main():
    mean=float(posterior_cash(MU[None,:],SIGMA,48)[0])
    second=float(posterior_cash(MU[None,:],SIGMA,48,True)[0])
    result={'prior_mean':mean,'prior_variance':second-mean*mean,'mse':{},
            'oracle_profit':{str(o):oracle_profit(o) for o in (24,36,48)}}
    for j in SUBSETS:
        _,v=posterior(MU[None,:],j)
        values={str(o):second-cross_moment(SIGMA-v,o) for o in (20,28)}
        result['mse'][key(j)]=values
        if not j:
            assert abs(values['28']-(second-mean*mean))<1e-6
        print(key(j), values,flush=True)
    for j in SUBSETS:
        for k in range(3):
            if k not in j:
                larger=tuple(sorted(j+(k,)))
                assert result['mse'][key(j)]['28']>=result['mse'][key(larger)]['28']-1e-6
    result['max_order_refinement']=max(abs(v['20']-v['28']) for v in result['mse'].values())
    (ROOT/'gaussian_integrals.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
