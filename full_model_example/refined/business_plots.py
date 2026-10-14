# coding: utf-8
import os
os.environ.setdefault('MPLCONFIGDIR',os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mplconfig')))
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,ScalarFormatter
from run import model,calibration_draws
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text());m=model();flow=100000
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'savefig.bbox':'tight'})
fmt=FuncFormatter(lambda x,pos:f'{x:,.0f}'.replace(',',' '))
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
f=PchipInterpolator([q['fitted_tau'] for q in r['curve']],[q['gain'] for q in r['curve']])
rng=np.random.default_rng(r['design']['seed'])
for row in r['by_N']:
 _,th=calibration_draws(m,row['N'],200000,rng)
 if row['N']==64:break
g=f(th)*flow
assert abs(g.mean()-r['primary']['mean_gain']*flow)<1e-8
fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
counts,edges=np.histogram(g,bins=np.unique(np.r_[np.linspace(g.min(),g.max(),61),0.])); centers=(edges[1:]+edges[:-1])/2
ax.bar(centers,counts/len(g)*100,width=np.diff(edges)*.95,color=np.where(centers<0,'#c74d47','#2a8c70'))
ax.axvline(0,color='#222');ax.axvline(g.mean(),color='#245d87',ls='--',label=f'Средний эффект: {g.mean():,.0f}'.replace(',',' '))
ax.set_title('Один и тот же скор: распределение денежных результатов обучения',loc='left',fontsize=12,fontweight='bold')
ax.set_xlabel('Изменение прибыли на 100 000 кандидатов, попугаи');ax.set_ylabel('Доля калибровок, %');ax.xaxis.set_major_formatter(fmt)
ax.text(.03,.9,f'Убыток в {100*np.mean(g<0):.1f}% пилотов\nN = 64; 200 000 калибровок',transform=ax.transAxes,va='top')
ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,.7));save(fig,'profit_distribution')

# Compare reference precision at the same customer economy and unchanged score.
rows=[];fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
for dh,color in [(1,'#2a8c70'),(4,'#245d87'),(16,'#c74d47')]:
 m.noise[0,0]=dh;rng=np.random.default_rng(2026101300+dh);ys=[];ses=[]
 for q in r['by_N']:
  _,th=calibration_draws(m,q['N'],200000,rng);g=f(th)*flow
  ys.append(float(g.mean()));ses.append(float(g.std(ddof=1)/np.sqrt(len(g))));rows.append(dict(N=q['N'],D_H=dh,mean_gain_per_100k=ys[-1],MCSE_per_100k=ses[-1]))
 ns=[q['N'] for q in r['by_N']];ys=np.array(ys);ses=np.array(ses)
 ax.plot(ns,ys,color=color,lw=2,label=f'Шум референса: σ = {int(np.sqrt(dh))}')
 ax.fill_between(ns,ys-1.96*ses,ys+1.96*ses,color=color,alpha=.15)
ax.axhline(0,color='#222',lw=.9);ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512,1024,2048,4096]);ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel('Клиентов в калибровке нового скора');ax.set_ylabel('Средний эффект, попугаи\nна 100 000 кандидатов');ax.yaxis.set_major_formatter(fmt)
ax.set_title('Качество референсных данных меняет необходимый размер пилота',loc='left',fontsize=12,fontweight='bold');ax.legend(frameon=False,loc='lower right')
save(fig,'reference_quality');(OUT/'reference_quality.json').write_text(json.dumps(rows,indent=2)+'\n')
