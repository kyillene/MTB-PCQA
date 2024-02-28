import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# color palette
cp = ['#00bfc7', '#514bd3', '#e8871a', '#cc2481']

# load the metric score differences and the ground truth significance labels
pred = pd.read_csv('data/pcqm_prediction_differences_testset.csv')
gt_signfs = pd.read_csv('./data/intraSRC_signf.csv')

# categorize the pairs based on statistical significance results (i.e., different or similar)
non_signf_pairs = pred[gt_signfs.loc[:, 'signf'] == 0]
signf_pairs = pred[gt_signfs.loc[:, 'signf'] ** 2 == 1]

# categorize the pairs based on the direction of the difference (i.e., better or worse)
w = pred[gt_signfs.loc[:, 'signf'] == -1]
b = pred[gt_signfs.loc[:, 'signf'] == 1]
bt, wt = b['pred_difference'] * -1, w['pred_difference'] * -1
b_inv, w_inv = b.copy(), w.copy()
b_inv['pred_difference'], w_inv['pred_difference'] = bt, wt
worse_pairs = pd.concat([w, b_inv], ignore_index=True)
better_pairs = pd.concat([b, w_inv], ignore_index=True)

# Plotting the different vs similar and better vs worse distributions

# get the range of the differences to set the x-axis limits and the width of the bars
msd = pred.loc[:, 'pred_difference'].to_numpy()
min_range = np.percentile(msd, 5)
max_range = np.percentile(msd, 95)
barw = np.max([min_range, max_range]) * 2 / 21
bar_sns = ((max_range - 0) / 20)

# create the figure and the axes
fig, ax = plt.subplots(1, 2, figsize=(20, 8), dpi=100)

# Plotting the different vs similar distribution

# get histogram of the absolute differences for similar pairs and plot the bar chart
hist, bins = np.histogram(np.abs(non_signf_pairs['pred_difference']), bins=20, range=[0, max_range])
mh = np.max(hist)  # temporary max value to set the y-axis limits
binshift = (bins[0] - bins[1]) / 8  # similar bars are shifted for the 1/8 of the bar width to prevent complete overlap with the "different" bars
ax[0].bar(bins[:-1] + binshift, hist, color=cp[1], alpha=0.7, width=0.6 * bar_sns, label='Similar')

# get histogram of the absolute differences for different pairs and plot the bar chart
hist, bins = np.histogram(np.abs(signf_pairs['pred_difference']), bins=20, range=[0, max_range])
ax[0].bar(bins[:-1], hist, color=cp[3], alpha=0.7, width=0.6 * bar_sns, label='Different')

# cosmetic changes
xticksvals = np.arange(0, np.max(np.abs(bins))*1.01, np.max(np.abs(bins))/5)
ax[0].set_xticks(xticksvals, fontsize=14)
ax[0].set_xlabel('Absolute Difference in Predicted Quality Score', fontsize=16)
ax[0].set_ylabel('Number of Pairs', fontsize=16)
hist_maxval = np.max([mh, np.max(hist)])  # max value for the y-axis limits
ax[0].set_ylim([0, hist_maxval*1.11])
ax[0].set_yticks(np.round(np.arange(0, hist_maxval*1.11, hist_maxval/11), 0), fontsize=14)
ax[0].grid(axis='y', linestyle='--', alpha=0.75)
ax[0].set_axisbelow(True)
ax[0].legend(loc='upper right', fontsize=16)
ax[0].set_title('Different vs Similar', fontsize=16)

# Plotting the better vs worse distribution

# get histogram of the differences for worse pairs and plot the bar chart
hist, bins = np.histogram(worse_pairs['pred_difference'], bins=21,
                          range=[-1 * np.max([min_range, max_range]), np.max([min_range, max_range])])
binshift = (bins[0] - bins[1]) / 8
ax[1].bar(bins[:-1] + binshift, hist, color=cp[1], alpha=0.75, width=0.6 * barw, label='Worse')

# get histogram of the differences for better pairs and plot the bar chart
hist, bins = np.histogram(better_pairs['pred_difference'], bins=21,
                          range=[-1 * np.max([min_range, max_range]), np.max([min_range, max_range])])
ax[1].bar(bins[:-1], hist, color=cp[3], alpha=0.75, width=0.6 * barw, label='Better')

# cosmetic changes
ax[1].set_xlabel('Difference in Predicted Quality Score', fontsize=16)
ax[1].set_ylabel('Number of Pairs', fontsize=16)
xticksvals = np.arange(-1 * np.max([min_range, max_range]), 1.01 * np.max([min_range, max_range]),
                       np.max([min_range, max_range]) / 3)
ax[1].set_xticks(xticksvals, fontsize=14)
hist_maxval = np.max(hist) # max value for the y-axis limits
ax[1].set_ylim([0, hist_maxval*1.11])
ax[1].set_yticks(np.round(np.arange(0, hist_maxval*1.11, hist_maxval/11),0), fontsize=14)
ax[1].vlines(x=xticksvals[3], ymin=0, ymax=hist_maxval*1.11, color='k', linestyles='--', linewidth=3, alpha=1)
ax[1].grid(axis='y', linestyle='--', alpha=0.75)
ax[1].set_axisbelow(True)
ax[1].legend(loc='upper right', fontsize=16)
ax[1].set_title('Better vs Worse', fontsize=16)
plt.show()