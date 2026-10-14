# coding: utf-8
import os
os.environ.setdefault('MPLCONFIGDIR',os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mplconfig')))
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.interpolate import PchipInterpolator

OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text()); flow=r['design']['flow']; p=r['primary'];base=r['baseline']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.17,'savefig.bbox':'tight'})
fmt=FuncFormatter(lambda x,pos:f'{x:,.0f}'.replace(',',' '))
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)

a=r['curve'];t=np.array([q['fitted_tau'] for q in a]);g=flow*np.array([q['gain'] for q in a]); tau=r['design']['tau']
fig,(ax,bx)=plt.subplots(2,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
ax.plot(t,g,color='#245d87',lw=2.5,label='Полная нелинейная модель')
ax.fill_between(t,g,0,where=g<0,color='#c74d47',alpha=.15)
ax.fill_between(t,g,0,where=g>=0,color='#2a8c70',alpha=.25)
ax.axhline(0,color='#333',lw=.9);ax.axvline(tau,color='#2a8c70',ls='--',label='Истинная сила скора τ = 0.25')
ax.scatter([tau],[flow*r['oracle_gain']],color='#2a8c70',zorder=4)
ax.set_title('Полезный скор может ухудшить решение при ошибочной калибровке',loc='left',fontsize=13)
ax.set_ylabel('Изменение прибыли, попугаи\nна 100 000 кандидатов');ax.yaxis.set_major_formatter(fmt);ax.legend(loc='lower right',frameon=False,fontsize=10)
h=r['pilot_distribution'];edges=np.array(h['edges']);counts=np.array(h['counts'])/sum(h['counts'])*100
bx.bar((edges[:-1]+edges[1:])/2,counts,width=np.diff(edges)*.93,color='#8297aa')
bx.axvline(tau,color='#2a8c70',ls='--');bx.set_ylabel('Доля обучений, %');bx.set_xlabel('Оценённая сила скора, τ̂')
bx.set_title('Как калибруется тот же скор по пилоту из 64 клиентов',loc='left',fontsize=11)
save(fig,'calibration_mechanism')

info=flow*r['oracle_gain'];cost=info-flow*p['mean_gain'];net=flow*p['mean_gain']
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
ax.bar(0,info,color='#2a8c70',width=.6)
ax.bar(1,-cost,bottom=info,color='#c74d47',width=.6)
ax.bar(2,net,color='#245d87',width=.6)
ax.plot([.3,.7],[info,info],color='#777',lw=1,ls=':');ax.plot([1.3,1.7],[net,net],color='#777',lw=1,ls=':')
for x,y,label in [(0,info,f'+{info:,.0f}'),(1,net/2,f'−{cost:,.0f}'),(2,net,f'{net:,.0f}')]:
    ax.annotate(label.replace(',',' '),(x,y),xytext=(0,9 if x==0 else -20 if x==2 else 0),textcoords='offset points',ha='center',fontweight='bold',fontsize=14)
ax.axhline(0,color='#333',lw=.9);ax.set_xticks([0,1,2],['Польза информации\nпри верной калибровке','Потеря относительно\nверной калибровки','Средний итог\nвключения скора'])
ax.set_ylabel('Попугаи на 100 000 кандидатов');ax.yaxis.set_major_formatter(fmt)
ax.set_title('Сигнал полезен, но его оценивание обходится дороже',loc='left',fontsize=13)
ax.set_ylim(min(net*1.45,-1100),info*1.5)
save(fig,'profit_decomposition')

rows=r['by_N'];ns=np.array([q['N'] for q in rows]);means=flow*np.array([q['mean_gain'] for q in rows]);ses=flow*np.array([q['mc_se'] for q in rows]);pw=np.array([q['fraction_worse'] for q in rows]);seprob=np.sqrt(pw*(1-pw)/200000)
fig,(ax,bx)=plt.subplots(2,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
ax.plot(ns,means,color='#245d87',lw=2,label='Средний эффект обучения')
ax.fill_between(ns,means-1.96*ses,means+1.96*ses,color='#245d87',alpha=.25,label='95% MC-интервал среднего')
ax.axhline(info,color='#2a8c70',ls='--',label='Польза при известной калибровке');ax.axhline(0,color='#333',lw=.9);ax.scatter([64],[net],color='#c74d47',zorder=4)
ax.set_ylabel('Изменение прибыли, попугаи\nна 100 000 кандидатов');ax.yaxis.set_major_formatter(fmt)
ax.set_title('Больше данных об одном и том же скоре меняет баланс',loc='left',fontsize=13);ax.legend(loc='lower right',frameon=False,fontsize=9)
bx.plot(ns,pw*100,color='#c74d47',lw=2)
bx.fill_between(ns,100*(pw-1.96*seprob),100*(pw+1.96*seprob),color='#c74d47',alpha=.25)
bx.set_ylabel('Обучения с потерей\nприбыли, %');bx.set_xlabel('Клиентов в пилоте для калибровки нового скора, N')
bx.set_ylim(0,100);bx.set_xscale('log',base=2);bx.set_xticks([32,64,128,256,512,1024,2048,4096]);bx.xaxis.set_major_formatter(ScalarFormatter())
fig.get_layout_engine().set(h_pad=.12)
save(fig,'learning_curve')

metrics={'baseline_profit_per_100k':flow*base['profit'],'expanded_profit_per_100k':flow*p['profit'],'oracle_profit_per_100k':flow*(base['profit']+r['oracle_gain']),'loss_per_100k':-flow*p['mean_gain'],'relative_profit_loss_pct':-100*p['mean_gain']/base['profit'],'baseline_roi_pct':100*base['roi'],'expanded_roi_pct':100*p['roi_ratio_of_expectations'],'roi_change_pp':100*(p['roi_ratio_of_expectations']-base['roi']),'baseline_acquired':flow*base['acquisition_probability'],'expanded_acquired':flow*(base['acquisition_probability']+p['mean_delta_accept']),'baseline_cac':flow*base['cac_spend'],'expanded_cac':flow*p['cac_spend'],'oracle_gain_per_100k':info,'estimation_loss_per_100k':cost,'mean_effect_MC_95_halfwidth':1.96*flow*p['mc_se'],'fraction_worse_pct':100*p['fraction_worse'],'effect_q05_q95_per_100k':[flow*p['q05'],flow*p['q95']],'max_checked_gain_error_per_100k':flow*max(abs(q['gain_error']) for q in r['checks'])}
if (OUT/'numerical_validation.json').exists():
    val=json.loads((OUT/'numerical_validation.json').read_text())
    metrics['max_midpoint_validation_error_per_100k']=flow*val['max_abs_gain_error']
(OUT/'business_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
