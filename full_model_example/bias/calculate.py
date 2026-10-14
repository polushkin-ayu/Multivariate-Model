import json
import numpy as np
from scipy.stats import norm, chi2
from numpy.polynomial.legendre import leggauss
from pathlib import Path

OUT=Path(__file__).resolve().parent
TAU=.25; VAR_H=.55**2

def compute(n,noise,order):
    df=n-1;vr=VAR_H+noise/2;vx=1+TAU**2*VAR_H;c=TAU*VAR_H
    x,w=leggauss(order);a=chi2.ppf((x+1)/2,df)
    mu=TAU*a/df;sd=np.sqrt(vr*vx-c*c)*np.sqrt(a)/(df*VAR_H)
    lo=(-1-mu)/sd;hi=(1-mu)/sd
    inside=norm.cdf(hi)-norm.cdf(lo)
    mean=mu*inside+sd*(norm.pdf(lo)-norm.pdf(hi))-norm.cdf(lo)+norm.sf(hi)
    clipprob=norm.cdf(lo)+norm.sf(hi)
    return dict(N=n,D_H=noise,raw_bias=0.,raw_sd=float(np.sqrt((vr*vx+c*c)/(df*VAR_H**2))),clipped_mean=float(w@mean/2),clipped_bias=float(w@mean/2-TAU),clipped_probability=float(w@clipprob/2))
rows=[]
for dh in [1,4,16]:
 for n in [64,512,4096]:
    a=compute(n,dh,512);b=compute(n,dh,1024);b['quadrature_mean_shift']=b['clipped_mean']-a['clipped_mean'];rows.append(b)
(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
