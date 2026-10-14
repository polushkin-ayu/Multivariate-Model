"""Analytic checks and bilingual numbers from saved results; no simulations."""
from pathlib import Path
import csv, json, math
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'generated'
OUT.mkdir(exist_ok=True)

def build():
    data = json.loads((ROOT/'full_model_example/refined/results.json').read_text())
    sigma, tau, dh, k = .55**2, .25, 4., 2
    s2 = (sigma + dh/k)/sigma**2
    curve = data['by_N']
    crossing = [(a['N'],b['N']) for a,b in zip(curve,curve[1:])
                if a['mean_gain'] < 0 <= b['mean_gain']]
    assert len(crossing) == 1
    rows=[]
    for row in data['by_N']:
        if row['N'] not in (64,512,4096): continue
        n=row['N']; h=tau*math.sqrt(n); q=h/math.sqrt(s2)
        rows.append(dict(N=n,h=h,h_over_s=q,local_loss_pct=200*NormalDist().cdf(-q),
                         local_gain_ratio=1-s2/h**2,full_gain_ratio=row['mean_gain']/data['oracle_gain'],
                         full_loss_pct=100*row['fraction_worse'],
                         full_gain_per_100000=row['mean_gain']*100000))
    baseline_s=math.sqrt((sigma+1/k)/sigma**2)
    raw_var=( (sigma+dh/k)*(1+tau*tau*sigma)+(tau*sigma)**2)/(63*sigma**2)
    gate_halfwidth=3*math.sqrt(.65**2+.75**2)
    coverage={str(mult):2*NormalDist().cdf(gate_halfwidth/math.sqrt(.65**2+mult*.75**2))-1
              for mult in (1,4,16)}
    result=dict(sigma_HH=sigma,tau=tau,s2=s2,s=math.sqrt(s2),baseline_s=baseline_s,
                N_star=s2/tau**2,crossing=crossing[0],rows=rows,raw_sd_N64=math.sqrt(raw_var),
                gate_coverage_WA=coverage,score_null_gate_coverage=2*NormalDist().cdf(3)-1,
                oracle_gain_per_100000=data['oracle_gain']*100000,
                estimator_fixed_tau_asymptotic_variance=raw_var*63)
    assert abs(baseline_s-2.961400474)<1e-9
    assert abs(math.sqrt(raw_var)-.63870)<5e-6
    assert round(result['oracle_gain_per_100000'])-11980 == -8711
    (OUT/'numbers.json').write_text(json.dumps(result,indent=2)+'\n')
    numbers={'BaselineS':baseline_s,'FixedS':result['s'],'LocalSampleThreshold':result['N_star'],
             'CrossingLowerN':crossing[0][0],'CrossingUpperN':crossing[0][1],
             'OracleGain':result['oracle_gain_per_100000'],'RawSdN64':result['raw_sd_N64']}
    for row in rows:
        for field in ('h_over_s','local_loss_pct','local_gain_ratio','full_gain_ratio','full_loss_pct','full_gain_per_100000'):
            numbers[field.replace('_','').title()+str(row['N'])]=row[field]
    with (OUT/'numbers_ru.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['key','value']);w.writerows((key,str(value).replace('.',',')) for key,value in numbers.items())
    print(json.dumps(result,indent=2))

if __name__=='__main__':build()
