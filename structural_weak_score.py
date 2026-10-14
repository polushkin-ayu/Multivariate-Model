"""Weak-score experiment for the original nonlinear cash function.

Deployment observes W=(activity, turnover) and X=tau*(H-mu_H)+eta,
eta~N(0,1). Calibration additionally observes independent reference repeats
of all three latent log parameters. References are NOT repeated ML seeds.
"""
import numpy as np
from scipy.stats import norm
from cac_calibration import GaussianCAC, fit_replicates
from cac_decision import BoundaryEngine, parameter_gradient


def require_diagonal_noise(model):
    if not np.allclose(model.noise,np.diag(np.diag(model.noise)),atol=1e-12):
        raise ValueError('this experiment requires diagonal reference noise')


def crossing_margin(model, tau=0.):
    require_diagonal_noise(model)
    j=[1,2]
    gain=np.linalg.solve((model.sigma+model.noise)[np.ix_(j,j)],model.sigma[j,:]).T
    _,q=model.posterior(np.zeros((1,2)),(1,2))
    loading=gain[:,1]-tau*tau*q[:,0]*gain[0,1]/(1+tau*tau*q[0,0])
    return float(loading[2]-abs(loading[0])-max(-loading[1],0.))


class PosteriorEvaluator:
    def __init__(self, cash, b, q):
        self.cash, self.b, self.q = cash, b, q

    def posterior(self, scores, subset):
        return self.b, self.q


def mean_cash(b, q, cash, order=24):
    evaluator = PosteriorEvaluator(cash, b, q)
    return GaussianCAC.predict(evaluator, np.empty((len(b), 0)), (), order)


def predict(model, w, x, tau, order=24):
    require_diagonal_noise(model)
    b, q = model.posterior(np.asarray(w), (1, 2))
    x = np.broadcast_to(np.asarray(x), (len(b),))
    direction = q[:, 0]
    denominator = 1 + tau * tau * q[0, 0]
    residual = x - tau * (b[:, 0] - model.mu[0])
    updated_b = b + (tau / denominator) * residual[:, None] * direction
    updated_q = q - (tau * tau / denominator) * np.outer(direction, direction)
    return mean_cash(updated_b, updated_q, model.cash, order)


def slope(model, w, order=24, step=1e-4):
    """Cov(G,H | W=w), by differentiating the Gaussian mean in Q[:,H]."""
    b, q = model.posterior(w, (1, 2))
    shift = step * q[:, 0]
    return (mean_cash(b + shift, q, model.cash, order)
            - mean_cash(b - shift, q, model.cash, order)) / (2 * step)


def fit(references, x, cash):
    model, diagnostics = fit_replicates(references, cash)
    # Fixed compact stabilization for the uniform-integrability argument.
    # Applicability requires the true parameter to lie in its interior.
    values, vectors = np.linalg.eigh(model.sigma)
    clipped_values = np.clip(values,1e-4,20.)
    sigma = (vectors*clipped_values)@vectors.T
    mu = np.clip(model.mu,-20.,20.)
    noise = np.diag(np.clip(np.diag(model.noise),1e-4,20.))
    stabilized = (not np.array_equal(values,clipped_values)
                  or not np.array_equal(mu,model.mu)
                  or not np.array_equal(noise,model.noise))
    model = GaussianCAC(mu,sigma,noise,cash)
    diagnostics['compact_stabilization_active'] = stabilized
    h_proxy = np.asarray(references).mean(axis=1)[:, 0]
    x = np.asarray(x)
    if x.shape != h_proxy.shape:
        raise ValueError('one weak score required per calibration client')
    covariance = np.cov(x, h_proxy, ddof=1)[0, 1]
    tau_raw = covariance / model.sigma[0, 0]
    # Explicit stabilization; asymptotically inactive in tau=h/sqrt(N).
    tau = float(np.clip(tau_raw, -1., 1.))
    r = references.shape[1]
    s2 = (model.sigma[0, 0] + model.noise[0, 0] / r) / model.sigma[0, 0]**2
    diagnostics.update(tau_raw=float(tau_raw), tau_clipped=bool(tau != tau_raw))
    return model, tau, float(np.sqrt(s2)), diagnostics


def soft_weight(tau, standard_error, alpha=.05):
    if not 0 < alpha < 1 or standard_error < 0:
        raise ValueError('invalid confidence parameters')
    if tau == 0:
        return 0.
    return float(max(0., 1 - norm.ppf(1 - alpha / 2) * standard_error / abs(tau)))


def mixed_predict(model, w, x, tau, weight, gate=None, order=24):
    """A single global mixing weight, optionally restricted to a fixed gate.

    gate=(lower_activity_score, upper_activity_score, absolute_X_limit).
    Outside the gate the forecast is exactly the base forecast.
    """
    if not 0 <= weight <= 1:
        raise ValueError('weight must be in [0,1]')
    w, x = np.asarray(w), np.asarray(x)
    base = model.predict(w,(1,2),order)
    selected = np.ones(len(w),dtype=bool)
    if gate is not None:
        lower, upper, limit = gate
        if not lower < upper or limit <= 0:
            raise ValueError('invalid gate')
        selected = (w[:,0]>=lower)&(w[:,0]<=upper)&(abs(x)<=limit)
    if weight and np.any(selected):
        base[selected] += weight*(predict(model,w[selected],x[selected],tau,order)-base[selected])
    return base


def boundary_constants(model, price=25., outer_order=18, inner_order=28, gate=None):
    if price <= 0 or crossing_margin(model) <= 0:
        raise ValueError('requires a positive price and certified upward crossings')
    engine = BoundaryEngine(model, (1, 2), price, outer_order, inner_order)
    x_second = 1.
    if gate is not None:
        from numpy.polynomial.legendre import leggauss
        lower, upper, limit = gate
        nodes, weights = leggauss(outer_order)
        wa = lower+(upper-lower)*(nodes+1)/2
        variance = model.sigma[1,1]+model.noise[1,1]
        engine.w[:,0] = wa
        engine.weights = weights*(upper-lower)/2*norm.pdf((wa-model.mu[1])/np.sqrt(variance))/np.sqrt(variance)
        engine.conditional_mean = model.mu[2]+model.sigma[1,2]/variance*(wa-model.mu[1])
        engine.root = engine.threshold(model)
        x_second = 2*norm.cdf(limit)-1-2*limit*norm.pdf(limit)
    w = engine.w.copy()
    w[:, engine.vcol] = engine.root
    eps = 1e-4
    derivative = (engine.predict_at(model, engine.root + eps)
                  - engine.predict_at(model, engine.root - eps)) / (2 * eps)
    if np.any(derivative <= 0):
        raise ValueError('nonpositive numerical boundary derivative')
    density = norm.pdf((engine.root - engine.conditional_mean) / engine.conditional_sd) / engine.conditional_sd
    measure = engine.weights * density / derivative
    a = slope(model, w, inner_order)
    grad = parameter_gradient(model, w, (1, 2), inner_order)
    matrix = .5 * np.einsum('i,ij,ik->jk', measure, grad, grad)
    return {'C': float(.5 * x_second * np.sum(measure * a*a)),
            'base_matrix': matrix, 'engine': engine}


class WeakBoundaryEngine:
    """Integrate policy profit DIFFERENCES under a true weak-score model.

Gaussian quadrature in (W_A,X); root integration in W_V avoids the high
variance of subtracting two sample profits. Upward crossings are assumed;
regularity certificates apply locally, not to all arbitrary fitted models.
"""
    def __init__(self, true, tau, price=25., outer_order=12, inner_order=24, gate=None):
        from cac_decision import normal_grid
        if price <= 0 or crossing_margin(true,tau) <= 0:
            raise ValueError('true model requires a certified positive-price boundary')
        self.true, self.tau, self.price, self.order = true, tau, price, inner_order
        load = np.array([[0, 1, 0], [0, 0, 1], [tau, 0, 0.]])
        cov = load @ true.sigma @ load.T + np.diag([true.noise[1, 1], true.noise[2, 2], 1.])
        means = np.array([true.mu[1], true.mu[2], 0.])
        indices = [0, 2]
        nuisance_cov = cov[np.ix_(indices, indices)]
        if gate is None:
            nodes, self.weights = normal_grid(2, outer_order)
            self.nuisance = means[indices] + nodes @ np.linalg.cholesky(nuisance_cov).T
        else:
            from numpy.polynomial.legendre import leggauss
            from scipy.stats import multivariate_normal
            lower, upper, limit = gate
            if not lower < upper or limit <= 0:
                raise ValueError('invalid gate')
            nodes, weights = leggauss(outer_order)
            wa = lower+(upper-lower)*(nodes+1)/2
            xx = limit*nodes
            aa,bb = np.meshgrid(wa,xx,indexing='ij')
            self.nuisance = np.column_stack([aa.ravel(),bb.ravel()])
            density = multivariate_normal.pdf(self.nuisance,mean=means[indices],cov=nuisance_cov)
            self.weights = (np.outer(weights,weights).ravel()*(upper-lower)*limit/2*density)
        gain = np.linalg.solve(nuisance_cov, cov[indices, 1])
        self.vmean = means[1] + (self.nuisance - means[indices]) @ gain
        self.vsd = np.sqrt(cov[1, 1] - cov[1, indices] @ gain)

    def forecast(self, model, tau, weight, v):
        w = np.column_stack([self.nuisance[:, 0], v])
        base = model.predict(w, (1, 2), self.order)
        if weight == 0:
            return base
        return base + weight * (predict(model, w, self.nuisance[:, 1], tau, self.order) - base)

    def root(self, model, tau, weight):
        if crossing_margin(model) <= 0 or (weight and crossing_margin(model,tau) <= 0):
            raise ValueError('fitted crossings are not certified; numerical evaluation refused')
        lo, hi = self.vmean - 16*self.vsd, self.vmean + 16*self.vsd
        if np.any(self.forecast(model,tau,weight,lo) >= self.price) or np.any(self.forecast(model,tau,weight,hi) <= self.price):
            raise ValueError('root not bracketed')
        for _ in range(36):
            mid = (lo + hi)/2
            above = self.forecast(model,tau,weight,mid) >= self.price
            hi, lo = np.where(above,mid,hi), np.where(above,lo,mid)
        return (lo + hi)/2

    def gain(self, model, fitted_tau, weight, base_root=None, legendre_order=10):
        from numpy.polynomial.legendre import leggauss
        if weight == 0:
            return 0.
        base_root = self.root(model,0.,0.) if base_root is None else base_root
        candidate = self.root(model,fitted_tau,weight)
        nodes, weights = leggauss(legendre_order)
        v = candidate[:,None] + (base_root-candidate)[:,None]*(nodes+1)/2
        w = np.column_stack([np.repeat(self.nuisance[:,0],legendre_order), v.ravel()])
        x = np.repeat(self.nuisance[:,1],legendre_order)
        payoff = predict(self.true,w,x,self.tau,self.order).reshape(v.shape)-self.price
        density = norm.pdf((v-self.vmean[:,None])/self.vsd)/self.vsd
        integral = np.sum(payoff*density*weights,axis=1)*(base_root-candidate)/2
        return float(self.weights @ integral)
