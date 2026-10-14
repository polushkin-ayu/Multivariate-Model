"""Profit loss, boundary coefficients and honest validation for noisy CAC scores.

Boundary integration is restricted to subsets containing turnover. It assumes
one upward crossing of the requested price in the turnover coordinate. Bracket
and derivative checks are numerical diagnostics, not a proof of this assumption.
All finite differences hold the observed score vector fixed.
"""
from itertools import product
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.stats import chi2, t as student_t, norm
from cac_calibration import GaussianCAC, moment_asymptotic_covariance

PAIRS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))


def pack(model):
    return np.r_[model.mu,[model.sigma[i,j] for i,j in PAIRS],np.diag(model.noise)]


def unpack(theta,cash):
    sigma=np.zeros((3,3))
    for value,(i,j) in zip(theta[3:9],PAIRS):
        sigma[i,j]=sigma[j,i]=value
    return GaussianCAC(theta[:3],sigma,np.diag(theta[9:]),cash)


def parameter_gradient(model,w,subset,order=20,step=2e-4):
    theta=pack(model)
    out=np.empty((len(w),len(theta)))
    for j in range(len(theta)):
        delta=np.zeros_like(theta)
        delta[j]=step*max(abs(theta[j]),.1)
        plus=unpack(theta+delta,model.cash).predict(w,subset,order)
        minus=unpack(theta-delta,model.cash).predict(w,subset,order)
        out[:,j]=(plus-minus)/(2*delta[j])
    return out


def normal_grid(dim,order):
    if dim==0:
        return np.zeros((1,0)),np.ones(1)
    x,w=hermgauss(order)
    ix=np.array(list(product(range(order),repeat=dim)))
    return np.sqrt(2)*x[ix],np.prod(w[ix]/np.sqrt(np.pi),axis=1)


def regularity_certificate(model,subset):
    """Sufficient global positive-price regularity in the turnover direction.

    |L_n'| <= L_n and 0 <= p' <= p imply upward crossings whenever
    loading_V > abs(loading_H) + max(-loading_A,0). See Proposition 3.
    A failure means no certificate, not necessarily an irregular boundary.
    """
    j=list(subset)
    if 2 not in j:
        raise ValueError('requires turnover in subset')
    gain=np.linalg.solve((model.sigma+model.noise)[np.ix_(j,j)],model.sigma[j,:]).T
    loading=gain[:,j.index(2)]
    margin=float(loading[2]-abs(loading[0])-max(-loading[1],0))
    return {'loading':loading.tolist(),'margin':margin,'certified_for_positive_prices':margin>0}


class BoundaryEngine:
    """Integrate over nuisance scores, solving for the turnover threshold.

    The nuisance grid and score density always come from the true/evaluation
    population, including when a differently calibrated policy is evaluated.
    """
    def __init__(self,model,subset,price,outer_order=16,inner_order=20):
        self.model=model
        self.subset=tuple(subset)
        if 2 not in subset:
            raise ValueError('boundary integrator requires a turnover score')
        self.price=float(price)
        if not np.isfinite(self.price) or self.price<0:
            raise ValueError('price must be finite and nonnegative')
        self.order=inner_order
        j=list(subset)
        self.vcol=j.index(2)
        self.other=[x for x in j if x!=2]
        self.ocol=[j.index(x) for x in self.other]
        covariance=model.sigma+model.noise
        x,self.weights=normal_grid(len(self.other),outer_order)
        self.w=np.zeros((len(x),len(j)))
        if self.other:
            c=covariance[np.ix_(self.other,self.other)]
            self.w[:,self.ocol]=model.mu[self.other]+x@np.linalg.cholesky(c).T
            gain=np.linalg.solve(c,covariance[self.other,2])
            self.conditional_mean=(model.mu[2]
                +(self.w[:,self.ocol]-model.mu[self.other])@gain)
            conditional_var=covariance[2,2]-covariance[2,self.other]@gain
        else:
            self.conditional_mean=np.full(len(x),model.mu[2])
            conditional_var=covariance[2,2]
        self.conditional_sd=np.sqrt(conditional_var)
        self.root=self.threshold(model)

    def predict_at(self,model,v):
        w=self.w.copy()
        w[:,self.vcol]=v
        return model.predict(w,self.subset,self.order)

    def threshold(self,model,iterations=34):
        low=self.conditional_mean-12*self.conditional_sd
        high=self.conditional_mean+12*self.conditional_sd
        for _ in range(4):
            bad_low=self.predict_at(model,low)>=self.price
            bad_high=self.predict_at(model,high)<=self.price
            if not np.any(bad_low|bad_high):
                break
            low=np.where(bad_low,self.conditional_mean+2*(low-self.conditional_mean),low)
            high=np.where(bad_high,self.conditional_mean+2*(high-self.conditional_mean),high)
        else:
            raise ValueError('upward crossing not bracketed after range expansion')
        for _ in range(iterations):
            middle=(low+high)/2
            above=self.predict_at(model,middle)>=self.price
            high=np.where(above,middle,high)
            low=np.where(above,low,middle)
        return (low+high)/2

    def crossing_diagnostics(self,model=None,grid=65):
        model=model or self.model
        signs=[]
        for x in np.linspace(-12,12,grid):
            signs.append(self.predict_at(model,self.conditional_mean+x*self.conditional_sd)>=self.price)
        signs=np.array(signs,dtype=int)
        changes=np.diff(signs,axis=0)
        return {'max_upward_crossings':int((changes>0).sum(axis=0).max()),
                'max_downward_crossings':int((changes<0).sum(axis=0).max())}

    def coefficient(self,repeats=2,step=2e-4):
        w=self.w.copy()
        w[:,self.vcol]=self.root
        grad=parameter_gradient(self.model,w,self.subset,self.order,step)
        covariance=moment_asymptotic_covariance(self.model.sigma,self.model.noise,repeats)
        variance=np.einsum('ni,ij,nj->n',grad,covariance,grad)
        eps=2e-4
        derivative=(self.predict_at(self.model,self.root+eps)
                    -self.predict_at(self.model,self.root-eps))/(2*eps)
        if np.any(derivative<=0):
            raise ValueError('nonpositive boundary derivative')
        standardized=(self.root-self.conditional_mean)/self.conditional_sd
        density=np.exp(-standardized**2/2)/(np.sqrt(2*np.pi)*self.conditional_sd)
        b=.5*np.sum(self.weights*density*variance/derivative)
        return {'B':float(b),'min_boundary_derivative':float(derivative.min()),
                'boundary_residual':float(np.max(np.abs(self.model.predict(w,self.subset,self.order)-self.price)))}

    def _integral(self,lower,upper,legendre_order):
        x,weights=leggauss(legendre_order)
        v=lower[:,None]+(upper-lower)[:,None]*(x+1)/2
        w=np.repeat(self.w,legendre_order,axis=0)
        w[:,self.vcol]=v.ravel()
        payoff=self.model.predict(w,self.subset,self.order).reshape(v.shape)-self.price
        t=(v-self.conditional_mean[:,None])/self.conditional_sd
        density=np.exp(-t*t/2)/(np.sqrt(2*np.pi)*self.conditional_sd)
        return np.sum(payoff*density*weights,axis=1)*(upper-lower)/2

    def loss(self,fitted,legendre_order=10):
        candidate=self.threshold(fitted)
        # The signed integral is nonnegative on either side of the oracle root.
        by_nuisance=self._integral(self.root,candidate,legendre_order)
        loss=float(self.weights@by_nuisance)
        if loss < -1e-7:
            raise ArithmeticError('negative loss: crossing or integration failure')
        return max(loss,0.)

    def oracle_profit(self,legendre_order=72):
        # Exponential-normal tail beyond 12 SD is negligible for reported cases;
        # this is a numerical truncation, not an exact integration guarantee.
        upper=np.maximum(self.conditional_mean+12*self.conditional_sd,self.root+8*self.conditional_sd)
        return float(self.weights@self._integral(self.root,upper,legendre_order))


def estimate_increment(model,base_subset,new_subset,price,calibration_clients,
                       repeats=2,outer_order=16,inner_order=24):
    """Plug-in asymptotic profit difference, not a confidence bound.

    When model is fitted, both the population law and B are estimated. Use an
    independent monetary holdout for a finite-sample validation guarantee.
    """
    if not set(base_subset)<set(new_subset) or calibration_clients<2:
        raise ValueError('require nested distinct subsets and N>=2')
    if min(np.linalg.eigvalsh(model.sigma).min(),np.diag(model.noise).min())<1e-4:
        return {'status':'insufficient_data','reason':'calibration_near_covariance_boundary'}
    if price<=0 or not all(regularity_certificate(model,j)['certified_for_positive_prices']
                          for j in (base_subset,new_subset)):
        return {'status':'insufficient_data','reason':'regularity_not_certified'}
    engines=[BoundaryEngine(model,j,price,outer_order,inner_order) for j in (base_subset,new_subset)]
    profits=[e.oracle_profit() for e in engines]
    coefficients=[e.coefficient(repeats)['B'] for e in engines]
    information=profits[1]-profits[0]
    penalty=(coefficients[1]-coefficients[0])/calibration_clients
    return {'status':'point_estimate_only','information_value':information,
            'base_B':coefficients[0],'new_B':coefficients[1],
            'additional_calibration_loss':penalty,
            'estimated_profit_difference':information-penalty,
            'N':calibration_clients,'price':price}


def cash_second_moment_bound(model,price):
    """Known-parameter upper bound on E[(Y-price)^2], including behavior noise.

    |Y-price| <= n*m*exp(V)+n*c+price for price>=0.
    """
    if not np.isfinite(price) or price<0:
        raise ValueError('price must be nonnegative')
    n,m,c=model.cash.horizon,model.cash.margin,model.cash.opex
    a=n*m
    b=n*c+price
    mu,var=model.mu[2],model.sigma[2,2]
    return float(a*a*np.exp(2*mu+2*var)+2*a*b*np.exp(mu+var/2)+b*b)


def cash_second_moment_upper_from_repeats(repeats,cash,price,beta=.025):
    """Finite-sample upper confidence bound on E[(Y-price)^2].

    Gaussian iid client averages imply exact one-sided Student and chi-square
    bounds. Var(average log-turnover) >= Var(true log-turnover) under independent,
    zero-mean measurement noise. A union bound gives confidence >=1-beta.
    This deliberately conservative bound does not pretend estimated D is known.
    """
    repeats=np.asarray(repeats,float)
    if repeats.ndim!=3 or repeats.shape[2]!=3 or len(repeats)<3 or repeats.shape[1]<1:
        raise ValueError('expected [N>=3, R>=1, 3] log-scores')
    if not np.isfinite(repeats).all() or not 0<beta<1 or not np.isfinite(price) or price<0:
        raise ValueError('invalid inputs')
    v=repeats[:,:,2].mean(axis=1)
    n=len(v); variance=v.var(ddof=1)
    mu_upper=float(v.mean()+student_t.ppf(1-beta/2,n-1)*np.sqrt(variance/n))
    variance_upper=float((n-1)*variance/chi2.ppf(beta/2,n-1))
    a=cash.horizon*cash.margin
    b=cash.horizon*cash.opex+price
    bound=float(a*a*np.exp(2*mu_upper+2*variance_upper)
                +2*a*b*np.exp(mu_upper+variance_upper/2)+b*b)
    return {'second_moment_upper':bound,'mu_v_upper':mu_upper,
            'sigma_vv_upper':variance_upper,'failure_probability':beta}


def validate_profit_difference(y,price,base_decision,new_decision,
                               second_moment_upper,alpha=.05):
    """Finite-sample Chebyshev interval for two *fixed* policies on iid holdout.

    The second-moment upper bound must be externally valid; plugging an
    estimated value into it does not preserve the guarantee. For multiple
    comparisons supply a Bonferroni-adjusted alpha. Returns an abstaining rule.
    """
    y=np.asarray(y,float)
    base=np.asarray(base_decision,bool)
    new=np.asarray(new_decision,bool)
    if y.ndim!=1 or base.shape!=y.shape or new.shape!=y.shape or len(y)<2:
        raise ValueError('need equally sized one-dimensional holdout arrays')
    if not np.isfinite(y).all() or not np.isfinite(price) or price<0 or not np.isfinite(second_moment_upper) or second_moment_upper<0 or not 0<alpha<1:
        raise ValueError('invalid finite observations, bound or alpha')
    values=(y-price)*(new.astype(int)-base.astype(int))
    mean=float(values.mean())
    radius=float(np.sqrt(second_moment_upper/(len(y)*alpha)))
    se=float(values.std(ddof=1)/np.sqrt(len(y)))
    normal_radius=float(norm.ppf(1-alpha/2)*se)
    return {'difference':mean,'lower':mean-radius,'upper':mean+radius,
            'decision':'add' if mean>radius else 'keep' if mean < -radius else 'insufficient_data',
            'alpha':alpha,'holdout_clients':len(y),
            'descriptive_se':se,
            'normal_approximation':{'lower':mean-normal_radius,'upper':mean+normal_radius,
                'coverage':'asymptotic only; not a finite-sample guarantee'}}
