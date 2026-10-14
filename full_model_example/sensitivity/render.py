# coding: utf-8
import os
os.environ.setdefault('MPLCONFIGDIR',os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mplconfig')))
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'refined_results.json').read_text());rows=r['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
for i,dh in enumerate([1,4,16]):
 for j,n in enumerate([64,512,4096]):
  ax=axes[i,j];data=np.empty((3,3));small=np.zeros((3,3),bool)
  for k,price in enumerate([25,75,150]):
   for l,d in enumerate([1,4,16]):
    z=next(z for z in rows if z['N']==n and z['D_H_reference_noise_multiplier']==dh and z['price']==price and z['D_A_multiplier']==d)
    data[k,l]=z['relative_mean_profit_percent'];small[k,l]=z['baseline_profit_per_100000']<1000
  im=ax.imshow(data,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),aspect='auto')
  for k in range(3):
   for l in range(3):
    text=f'{data[k,l]:+.2f}%'+('*' if small[k,l] else '')
    ax.text(l,k,text,ha='center',va='center',fontsize=9,color='white' if abs(data[k,l])>3 else '#222')
  ax.set_xticks([0,1,2],['×1','×2','×4']);ax.set_yticks([0,1,2],[25,75,150])
  ax.set_title(f'N = {n}; шум референса σ = {int(np.sqrt(dh))}',fontsize=10)
  if j==0:ax.set_ylabel('CAC, попугаи')
  if i==2:ax.set_xlabel('Шум базовых скоров: множитель σ')
fig.suptitle('Когда дополнительный скор уменьшает прибыль больше чем на 1%\nИзменение средней прибыли относительно базы; вся сетка из 81 сценария',fontsize=13,fontweight='bold')
fig.colorbar(im,ax=axes,shrink=.75,extend='both',label='Изменение прибыли, % (цвет ограничен диапазоном ±5%)')
fig.text(.5,-.01,'* Базовая прибыль менее 1000 попугаев на 100 000 кандидатов: большой процент чувствителен к знаменателю.',ha='center',fontsize=9)
fig.savefig(OUT/'profit_map.png',dpi=180);fig.savefig(OUT/'profit_map.pdf')
