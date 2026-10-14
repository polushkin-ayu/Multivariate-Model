import json
import numpy as np
from run import OUT, model, WeakBoundaryEngine, TAU, PRICE, GATE, effects, PchipInterpolator
m=model();r=json.loads((OUT/'results.json').read_text())
e=WeakBoundaryEngine(m,TAU,PRICE,28,36,gate=GATE);base=e.root(m,0,0)
grid=np.array([q['fitted_tau'] for q in r['curve']]); mids=(grid[:-1]+grid[1:])/2
inter={key:PchipInterpolator(grid,[q[key] for q in r['curve']]) for key in ['gain','delta_accept']}
rows=[]
for ix in np.linspace(0,len(mids)-1,17).astype(int):
    t=float(mids[ix]);hi=effects(e,m,t,base,vertical_order=32)
    rows.append(dict(tau=t,gain_high=hi[0],accept_high=hi[1],gain_error=float(inter['gain'](t)-hi[0]),accept_error=float(inter['delta_accept'](t)-hi[1])))
    print('validated',len(rows),'/17',flush=True)
out=dict(note='Simultaneous refinement of outer, inner and boundary-interval orders at 17 fixed grid midpoints; diagnostic, not a certified bound.',checks=rows,max_abs_gain_error=max(abs(q['gain_error']) for q in rows),max_abs_accept_error=max(abs(q['accept_error']) for q in rows),baseline_profit_shift=r['baseline']['profit']-r['baseline_order_check']['profit'],baseline_accept_shift=r['baseline']['acquisition_probability']-r['baseline_order_check']['acquisition_probability'])
(OUT/'numerical_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
