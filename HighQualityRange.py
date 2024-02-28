import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as plcc
from scipy.stats import spearmanr as srocc
import scipy.optimize as opt


def f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b


gt = pd.read_csv('data/BASICS_testset_mos_std_ci.csv')
preds = pd.read_csv('data/pcqm_predictions.csv')
preds = preds[preds['ppc'].isin(gt['ppc'])]

a = opt.curve_fit(f, preds['predictions'], gt['mos'], method="trf")[0]
preds_mapping = f(preds['predictions'], a[0], a[1], a[2], a[3])
preds['predictions'] = preds_mapping

gt_lowq = gt[gt['mos'] < 3.5]  # filter stimuli with MOS < 3.5
preds_lowq = preds[preds['ppc'].isin(gt_lowq['ppc'])]

gt = gt[gt['mos'] >= 3.5]  # filter stimuli with MOS >= 3.5
preds = preds[preds['ppc'].isin(gt['ppc'])]

srocc_score = srocc(preds['predictions'], gt['mos'])
plcc_score = plcc(preds['predictions'], gt['mos'])

print('SROCC: ', srocc_score[0])
print('PLCC: ', plcc_score[0])

fig, ax = plt.subplots(figsize=(11, 10), dpi=200)

ax.scatter(preds['predictions'], gt['mos'], s=10, c='b', alpha=0.75, zorder=3)
ax.scatter(preds_lowq['predictions'], gt_lowq['mos'], s=10, c='r', alpha=0.25, zorder=3)  # low quality content not used in evaluation, just included in the plot.
ax.set_xlabel('Predicted Quality Score (with mapping)', fontsize=13)
ax.set_ylabel('MOS', fontsize=13)
ax.set_xlim([0.92, 5.08])
ax.set_xticks(np.arange(1, 5.1, 1), np.round(np.arange(1, 5.1, 1), 2).tolist())
ax.set_ylim([0.92, 5.08])
ax.set_yticks(np.arange(1, 5.1, 1), np.round(np.arange(1, 5.1, 1), 2).tolist())
ax.tick_params(axis='both', which='major', length=5, width=1)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
ax.grid(axis='both', linestyle='--', alpha=0.9, linewidth=1, zorder=0)
ax.set_title('SROCC: {} --- PLCC: {}'.format(np.round(srocc_score[0], 4), np.round(plcc_score[0], 4)))
plt.show()
