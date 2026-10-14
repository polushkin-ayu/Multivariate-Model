"""Build journal revision figures and summaries from cached model outputs only."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / 'refined/results.json').read_text())
REF = json.loads((ROOT / 'refined/reference_quality.json').read_text())
SENS = json.loads((ROOT / 'sensitivity/refined_results.json').read_text())
THEOREM = json.loads((ROOT / 'theorem_training/summary.json').read_text())
RAW = np.load(ROOT / 'theorem_training/paired_gains.npz')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12,
                     'axes.labelsize':10, 'legend.fontsize':9, 'pdf.fonttype':42})

def ru_amount_tick(value, _position):
    rounded = round(value)
    if abs(value - rounded) < 1e-8:
        label = f'{int(rounded):,}'.replace(',', ' ')
    else:
        label = f'{value:,.1f}'.replace(',', ' ').replace('.', ',')
    return label.replace('-', '−')


# Diagram: operational flow from estimated score error to the CAC decision.
def process(lang):
    ru = lang == 'ru'
    labels = (['Скоры и повторные\nизмерения', 'Оценка коэффициента\n$\\hat{\\tau}$',
               'Прогноз денежного\nпотока', 'Сравнение с CAC', 'Решение и прибыль'] if ru else
              ['Scores and repeated\nmeasurements', 'Estimate coefficient\n$\\hat{\\tau}$',
               'Forecast cash\nflow', 'Compare with CAC', 'Decision and profit'])
    fig, ax = plt.subplots(figsize=(12.0, 2.55))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.55); ax.axis('off')
    xs = [0.25, 2.65, 5.05, 7.45, 9.85]
    widths = [1.8, 1.8, 1.8, 1.8, 1.8]
    colors = ['#EAF1F8','#E6F4F1','#FFF3DB','#FCE8E6','#E8F0FE']
    for i,(x,w,label,color) in enumerate(zip(xs,widths,labels,colors)):
        patch = FancyBboxPatch((x,.92),w,.95,boxstyle='round,pad=0.08,rounding_size=0.12',
                               facecolor=color,edgecolor='#536579',linewidth=1.2)
        ax.add_patch(patch); ax.text(x+w/2,1.395,label,ha='center',va='center',fontsize=10.2)
        if i < len(xs)-1:
            ax.add_patch(FancyArrowPatch((x+w+.06,1.395),(xs[i+1]-.08,1.395),
                          arrowstyle='-|>',mutation_scale=14,color='#536579',linewidth=1.3))
    note = 'Шум скоров, шум измерений, $N$' if ru else 'Score noise, measurement noise, $N$'
    ax.text(xs[0]+widths[0]/2,.48,note,ha='center',va='center',color='#526273',fontsize=8.0)
    fig.tight_layout(pad=.2)
    stem=f'process_{lang}'; fig.savefig(OUT/(stem+'.png'),dpi=220,bbox_inches='tight'); fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight'); plt.close(fig)
process('ru'); process('en')

# Theorem validation: right panel's mean normalized absolute residual only, from saved cases.
C_s2 = THEOREM['C_gate'] * THEOREM['s']**2
fig, ax = plt.subplots(figsize=(6.6,4.5))
markers={0.5:'o',2.0:'s'}; colors={0.5:'#555555',2.0:'#111111'}
for ratio in (0.5,2.0):
    cases=sorted([c for c in THEOREM['cases'] if c['h_over_s']==ratio],key=lambda c:c['N'])
    ax.plot([c['N'] for c in cases],[c['mean_absolute_residual_normalized_Cs2'] for c in cases],
            marker=markers[ratio],color=colors[ratio],lw=1.8,label=f'$h/s={str(ratio).replace(".", ",")}$')
ax.set(xscale='log',yscale='log',xlabel='Размер калибровки $N$' if True else 'N',
       ylabel='Среднее $|N\\Delta\\Pi-C(2hT-T^2)|/(Cs^2)$',
       title='Среднее нормированное $L_1$-расхождение')
ax.grid(alpha=.23,which='both'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'theorem_mean_l1_residual_ru.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'theorem_mean_l1_residual_ru.pdf',bbox_inches='tight'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.6,4.5))
for ratio in (0.5,2.0):
    cases=sorted([c for c in THEOREM['cases'] if c['h_over_s']==ratio],key=lambda c:c['N'])
    ax.plot([c['N'] for c in cases],[c['mean_absolute_residual_normalized_Cs2'] for c in cases],
            marker=markers[ratio],color=colors[ratio],lw=1.8,label=f'$h/s={ratio:g}$')
ax.set(xscale='log',yscale='log',xlabel='Calibration sample size $N$',
       ylabel='Mean $|N\\Delta\\Pi-C(2hT-T^2)|/(Cs^2)$',
       title='Mean normalized $L_1$ residual')
ax.grid(alpha=.23,which='both'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'theorem_mean_l1_residual_en.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'theorem_mean_l1_residual_en.pdf',bbox_inches='tight'); plt.close(fig)

# Two-panel learning and reference-quality figure, using the existing 38-point N grid.
by_n={r['N']:r for r in RESULTS['by_N']}
ref_by_dh={d:{r['N']:r for r in REF if r.get('D_H')==d} for d in (1,4,16)}
Ns=sorted(set(by_n)&set(ref_by_dh[1])&set(ref_by_dh[4])&set(ref_by_dh[16]))
flow=RESULTS['design']['flow']
for lang in ('ru','en'):
    fig,(left,right)=plt.subplots(1,2,figsize=(12.6,4.9),constrained_layout=True)
    gain=np.array([by_n[n]['mean_gain']*flow for n in Ns])
    gain_se=np.array([by_n[n]['mc_se']*flow for n in Ns])
    left.errorbar(Ns,gain,yerr=1.96*gain_se,marker='o',ms=3,lw=1.5,capsize=2,
                  color='#2878B5',label='Ожидаемая денежная прибавка' if lang=='ru' else 'Expected monetary gain')
    left.axhline(0,color='#555',lw=.8)
    left.set_xscale('log'); left.set_xticks([32,64,128,256,512,1024,2048,4096]); left.set_xticklabels(['32','64','128','256','512','1 024','2 048','4 096'] if lang=='ru' else ['32','64','128','256','512','1,024','2,048','4,096'])
    left.set_xlabel('Размер калибровки N' if lang=='ru' else 'Calibration sample size N')
    left.set_ylabel('Денежный эффект, ден. ед. на 100 000' if lang=='ru' else 'Monetary gain, currency units per 100,000')
    left.set_title('Один скор: объём выборки' if lang=='ru' else 'Fixed score: sample size')
    left.grid(alpha=.22); left.legend(frameon=False)
    dh_colors={1:'#2878B5',4:'#E1812C',16:'#3A923A'}
    for dh in (1,4,16):
        q=np.array([ref_by_dh[dh][n]['mean_gain_per_100k'] for n in Ns])
        se=np.array([ref_by_dh[dh][n]['MCSE_per_100k'] for n in Ns])
        right.errorbar(Ns,q,yerr=1.96*se,marker={1:'o',4:'s',16:'^'}[dh],ms=3,lw=1.35,capsize=2,
                       color=dh_colors[dh],label=f'$D_H={dh}$')
    right.axhline(0,color='#555',lw=.8)
    right.set_xscale('log'); right.set_xticks([32,64,128,256,512,1024,2048,4096]); right.set_xticklabels(['32','64','128','256','512','1 024','2 048','4 096'] if lang=='ru' else ['32','64','128','256','512','1,024','2,048','4,096'])
    right.set_xlabel('Размер калибровки N' if lang=='ru' else 'Calibration sample size N')
    right.set_ylabel('Денежный эффект, ден. ед. на 100 000' if lang=='ru' else 'Monetary gain, currency units per 100,000')
    right.set_title('Качество измерений' if lang=='ru' else 'Measurement quality')
    if lang == 'ru':
        left.yaxis.set_major_formatter(FuncFormatter(ru_amount_tick))
        right.yaxis.set_major_formatter(FuncFormatter(ru_amount_tick))
    right.grid(alpha=.22); right.legend(title='$D_H$',frameon=False)
    fig.suptitle('Денежный эффект и повторные измерения' if lang=='ru' else 'Monetary gain and repeated measurements',fontsize=13)
    fig.savefig(OUT/f'learning_reference_{lang}.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/f'learning_reference_{lang}.pdf',bbox_inches='tight'); plt.close(fig)

# Business table: profit, CAC spend and cash-before-CAC on 100k flow; ROI is ratio of expectations.
base=RESULTS['baseline']; price=RESULTS['design']['price']
rows=[]
for n in (64,512,4096):
    r=by_n[n]
    rows.append({'N':n,'expected_gain_per_100000':r['mean_gain']*flow,
                 'expected_profit_per_100000':r['profit']*flow,
                 'expected_acquisition_probability':base['acquisition_probability']+r['mean_delta_accept'],
                 'expected_acquired_per_100000':(base['acquisition_probability']+r['mean_delta_accept'])*flow,
                 'expected_cac_spend_per_100000':price*(base['acquisition_probability']+r['mean_delta_accept'])*flow,
                 'expected_cash_before_cac_per_100000':(base['cash_before_cac']+r['mean_gain']+r['mean_delta_accept']*price)*flow,
                 'roi_ratio_of_expectations':r['roi_ratio_of_expectations'],
                 'fraction_worse':r['fraction_worse'], 'mc_se_gain_per_100000':r['mc_se']*flow,
                 'baseline':{'expected_profit_per_100000':base['profit']*flow,
                             'expected_acquisition_probability':base['acquisition_probability'],
                             'expected_acquired_per_100000':base['acquisition_probability']*flow,
                             'expected_cac_spend_per_100000':base['cac_spend']*flow,
                             'expected_cash_before_cac_per_100000':base['cash_before_cac']*flow,
                             'roi_ratio_of_expectations':base['roi']}})
business={'units':'currency units and users per 100,000 eligible users; probability and ROI are ratios',
          'roi_definition':'expected profit / expected CAC spend (ratio of expectations)',
          'acquisition_probability_definition':'expected CAC spend / unit price',
          'source':'refined/results.json; no new simulations', 'rows':rows}
(OUT/'business_table.json').write_text(json.dumps(business,ensure_ascii=False,indent=2)+'\n')

# Sensitivity summary: the 81 prespecified grid rows are cases, not a population sample.
srows=SENS['rows']; neg=[r for r in srows if r['mean_delta_per_100000']<0]
loss=[r for r in srows if r['relative_mean_profit_percent'] < -1]
from collections import defaultdict
by_group=defaultdict(list)
for r in srows: by_group[str(r['N'])].append(r)
def agg(rs):
    return {'cases':len(rs),'negative_mean_count':sum(x['mean_delta_per_100000']<0 for x in rs),
            'loss_over_1_percent_count':sum(x['relative_mean_profit_percent'] < -1 for x in rs),
            'negative_mean_share_of_grid':sum(x['mean_delta_per_100000']<0 for x in rs)/len(rs) if rs else None,
            'loss_over_1_percent_share_of_grid':sum(x['relative_mean_profit_percent'] < -1 for x in rs)/len(rs) if rs else None}
sensitivity={'source':'sensitivity/refined_results.json','grid_cases':len(srows),
 'warning':'Counts and shares describe this prespecified 81-case grid only; they are not population frequencies.',
 'negative_mean_cases':len(neg),'negative_mean_share_of_grid':len(neg)/len(srows),
 'loss_over_1_percent_cases':len(loss),'loss_over_1_percent_share_of_grid':len(loss)/len(srows),
 'threshold_definition':'loss_over_1_percent means relative_mean_profit_percent < -1%',
 'source_MCSE_available':True,
 'classification':'Sign is classified by point estimate. MCSE intervals are retained per case below; small negative estimates are not described as robustly negative.',
 'by_N':{n:agg(rs) for n,rs in sorted(by_group.items(),key=lambda kv:int(kv[0]))},
 'cases':[{'N':r['N'],'D_A_multiplier':r['D_A_multiplier'],'D_V_multiplier':r['D_V_multiplier'],
           'price':r['price'],'true_tau':r['true_tau'],'D_H_reference_noise_multiplier':r['D_H_reference_noise_multiplier'],
           'mean_delta_per_100000':r['mean_delta_per_100000'],'MCSE_delta_per_100000':r['MCSE_delta_per_100000'],
           'delta_95pct_interval':[r['mean_delta_per_100000']-1.96*r['MCSE_delta_per_100000'],r['mean_delta_per_100000']+1.96*r['MCSE_delta_per_100000']],
           'relative_mean_profit_percent':r['relative_mean_profit_percent'],
           'classification_negative_mean':r['mean_delta_per_100000']<0,
           'classification_loss_over_1_percent':r['relative_mean_profit_percent'] < -1}
          for r in srows]}
(OUT/'sensitivity_summary.json').write_text(json.dumps(sensitivity,ensure_ascii=False,indent=2)+'\n')

# Copy existing original assets if available; preserve them byte-for-byte.
for name in ['profit_decomposition_ru.png','profit_decomposition_ru.pdf','profit_decomposition_en.png','profit_decomposition_en.pdf',
             'calibration_mechanism_ru.png','calibration_mechanism_ru.pdf','calibration_mechanism_en.png','calibration_mechanism_en.pdf']:
    src=ROOT/'refined'/name
    if src.exists(): shutil.copy2(src,OUT/name)

manifest={'inputs':['refined/results.json','refined/reference_quality.json','sensitivity/refined_results.json',
                    'theorem_training/summary.json','theorem_training/paired_gains.npz'],
          'simulation_policy':'No new simulations, scenarios, or theorem cases; figures and summaries use cached outputs only.',
          'outputs':sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'),
          'numeric_validation':{'business_table_N':[r['N'] for r in rows], 'learning_curve_N_count':len(Ns),
             'reference_rows_per_DH':{str(d):len(ref_by_dh[d]) for d in (1,4,16)},
             'reference_quality_DH': [1,4,16],
             'learning_left_source':'refined/results.json; base model; no new simulation',
             'learning_right_source':'refined/reference_quality.json; D_H=1,4,16; all existing rows','theorem_cases':len(THEOREM['cases']),
             'sensitivity_grid_cases':len(srows),'sensitivity_negative_point_estimate_count':len(neg),
             'sensitivity_loss_over_1_percent_count':len(loss),
             'roi_source_field':'roi_ratio_of_expectations',
             'validated_profit_identity_max_abs':max(abs((r['profit']-base['profit'])-r['mean_gain']) for r in RESULTS['by_N']),
             'validated_accept_identity_max_abs':max(abs((base['acquisition_probability']+r['mean_delta_accept'])-r['cac_spend']/price) for r in RESULTS['by_N']),
             'validated_roi_identity_max_abs':max(abs(r['roi_ratio_of_expectations']-r['profit']/r['cac_spend']) for r in RESULTS['by_N']),
             'validated_cash_identity_max_abs':max(abs((base['cash_before_cac']+r['mean_gain']+r['mean_delta_accept']*price)-(r['profit']+r['cac_spend'])) for r in RESULTS['by_N']),
             'monetary_gain_scaling':'mean_gain and mc_se multiplied by design.flow=100000',
             'sensitivity_MCSE_interval':'mean_delta_per_100000 ± 1.96 * MCSE_delta_per_100000'}}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
