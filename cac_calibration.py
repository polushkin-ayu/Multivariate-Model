"""A small empirical-Bayes CAC calibrator for repeated Gaussian log-scores.

fit_replicates takes [clients, independent_replicates, 3] log-score arrays.
Its independence assumption is essential: random seeds are not replicates
unless their prediction errors really satisfy this measurement model.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class CashSpec:
    horizon: int = 12
    margin: float = .1
    opex: float = 3.

    def __post_init__(self):
        if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if not np.isfinite(self.margin) or self.margin <= 0:
            raise ValueError('margin must be positive and finite')
        if not np.isfinite(self.opex) or self.opex < 0:
            raise ValueError('opex must be nonnegative and finite')


def _psd_sqrt(cov):
    vals,vecs=np.linalg.eigh((cov+cov.T)/2)
    if vals.min() < -1e-9:
        raise ValueError('covariance is not positive semidefinite')
    return vecs@np.diag(np.sqrt(np.maximum(vals,0)))


def _duration(h,n):
    lam=np.exp(h)
    return -np.expm1(-n*lam)/-np.expm1(-lam)


@dataclass
class GaussianCAC:
    mu: np.ndarray
    sigma: np.ndarray
    noise: np.ndarray
    cash: CashSpec = CashSpec()

    def __post_init__(self):
        self.mu=np.asarray(self.mu,dtype=float)
        self.sigma=np.asarray(self.sigma,dtype=float)
        self.noise=np.asarray(self.noise,dtype=float)
        if self.mu.shape!=(3,) or self.sigma.shape!=(3,3) or self.noise.shape!=(3,3):
            raise ValueError('expected three log-parameters: churn, activity, turnover')
        if not all(np.isfinite(x).all() for x in (self.mu,self.sigma,self.noise)):
            raise ValueError('parameters must be finite')
        for name,matrix in [('sigma',self.sigma),('noise',self.noise)]:
            if not np.allclose(matrix,matrix.T,atol=1e-12):
                raise ValueError(name+' must be symmetric')
            if np.linalg.eigvalsh(matrix).min()<=0:
                raise ValueError(name+' must be positive definite')

    def posterior(self,scores,subset=(0,1,2)):
        """scores has one row per client and only the columns in subset."""
        subset=tuple(subset)
        if len(set(subset))!=len(subset) or any(j not in range(3) for j in subset):
            raise ValueError('invalid score subset')
        w=np.asarray(scores,dtype=float)
        if w.ndim!=2 or w.shape[1]!=len(subset) or not np.isfinite(w).all():
            raise ValueError('scores must be a finite [clients, len(subset)] array')
        if not subset:
            return np.broadcast_to(self.mu,(len(w),3)).copy(),self.sigma.copy()
        j=list(subset)
        covariance=self.sigma[np.ix_(j,j)]+self.noise[np.ix_(j,j)]
        gain=np.linalg.solve(covariance,self.sigma[j,:]).T
        b=self.mu+(w-self.mu[j])@gain.T
        q=self.sigma-gain@self.sigma[j,:]
        return b,(q+q.T)/2

    def predict(self,scores,subset=(0,1,2),order=20):
        """Posterior mean of pre-acquisition cash, without clipping negatives."""
        if order<2:
            raise ValueError('quadrature order must be at least two')
        b,q=self.posterior(scores,subset)
        x,weights=hermgauss(order)
        ix=np.array(list(product(range(order),repeat=2)))
        nodes=np.sqrt(2)*x[ix]
        weights2=np.prod(weights[ix]/np.sqrt(np.pi),axis=1)
        offsets=nodes@_psd_sqrt(q[:2,:2]).T
        output=np.empty(len(b))
        for begin in range(0,len(b),512):
            sl=slice(begin,begin+512)
            center=b[sl]
            u=center[:,None,:2]+offsets+q[:2,2]
            la=_duration(u[:,:,0],self.cash.horizon)*(-np.expm1(-np.exp(u[:,:,1])))
            h=center[:,0,None]+np.sqrt(max(q[0,0],0))*np.sqrt(2)*x
            exposure=_duration(h,self.cash.horizon)@(weights/np.sqrt(np.pi))
            output[sl]=(self.cash.margin*np.exp(center[:,2]+q[2,2]/2)*(la@weights2)
                        -self.cash.opex*exposure)
        return output


def fit_replicates(replicates,cash=CashSpec(),diagonal_noise=True,eigen_floor=1e-8):
    """Moment estimates; covariance clipping is explicit in diagnostics.

    The same noise covariance is assumed for each replicate. Clients and
    replicate errors are independent, conditional on fixed client parameters.
    Uses all N*R scores to fit population parameters. The returned model still
    predicts from a SINGLE fresh score vector, whose noise covariance is D.
    """
    w=np.asarray(replicates,dtype=float)
    if w.ndim!=3 or w.shape[2]!=3 or w.shape[0]<2 or w.shape[1]<2:
        raise ValueError('need [N>=2, R>=2, 3] repeat log-scores')
    if not np.isfinite(w).all() or not np.isfinite(eigen_floor) or eigen_floor<=0:
        raise ValueError('inputs must be finite and eigen_floor positive')
    n,r,_=w.shape
    averages=w.mean(axis=1)
    mu=averages.mean(axis=0)
    centered=w-averages[:,None,:]
    within=np.einsum('nri,nrj->ij',centered,centered)/(n*(r-1))
    noise=np.diag(np.diag(within)) if diagonal_noise else within
    dn,un=np.linalg.eigh(noise)
    noise=(un*np.maximum(dn,eigen_floor))@un.T
    between=np.cov(averages,rowvar=False,ddof=1)
    raw_sigma=between-noise/r
    ds,us=np.linalg.eigh(raw_sigma)
    sigma=(us*np.maximum(ds,eigen_floor))@us.T
    diagnostics={
        'clients':n,'replicates':r,'diagonal_noise':diagonal_noise,
        'unrestricted_within_covariance':within.tolist(),
        'raw_sigma_eigenvalues':ds.tolist(),
        'noise_eigenvalues_before_floor':dn.tolist(),
        'sigma_repair_frobenius':float(np.linalg.norm(sigma-raw_sigma)),
        'repeat_mean_offsets':(w.mean(axis=0)-mu).tolist(),
    }
    return GaussianCAC(mu,sigma,noise,cash),diagnostics


def moment_asymptotic_covariance(sigma,noise,replicates=2):
    """Covariance of sqrt(N)*(theta_hat-theta), for diagonal-noise moments.

    theta ordering: mu[0:3], Sigma upper triangle (00,01,02,11,12,22),
    diag(D). Assumes an interior point: covariance projection is inactive
    asymptotically. Gaussian mean/covariance and within/between independence
    give the blocks below.
    """
    if replicates<2 or not np.allclose(noise,np.diag(np.diag(noise))):
        raise ValueError('requires R>=2 and diagonal noise')
    pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    b=sigma+noise/replicates
    dvar=2*np.diag(noise)**2/(replicates-1)
    result=np.zeros((12,12))
    result[:3,:3]=b
    result[9:,9:]=np.diag(dvar)
    for u,(i,j) in enumerate(pairs):
        for v,(k,l) in enumerate(pairs):
            result[3+u,3+v]=b[i,k]*b[j,l]+b[i,l]*b[j,k]
            if i==j==k==l:
                result[3+u,3+v]+=dvar[i]/replicates**2
        if i==j:
            result[3+u,9+i]=result[9+i,3+u]=-dvar[i]/replicates
    return result
