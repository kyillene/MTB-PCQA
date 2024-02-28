import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

rawscores = pd.read_csv('data/BASICS_testset_indv_scores.csv')
# the list of SRCs in the testset
srclist = rawscores['src'].unique()

# number of observers per PVS. If the nb of observers per PVS is not the same, the missing observers will be replaced
# by the median vote of each PVS.
nb_obs = 63  # this is the max number of observers per PVS in the BASICS dataset
intraSRCsignf = pd.DataFrame()
c = 0
for s in (srclist):
    print('SRC: ', s, ' --- ', c+1, ' / ', len(srclist))
    # filter out the stimuli that belong to the current SRC
    src_stims = rawscores[rawscores.loc[:, 'src'].str.contains(s)]
    pvsnames = src_stims.loc[:, 'ppc']  # get the PVS names
    votes = src_stims.iloc[:, 2:].to_numpy()  # get the votes
    # replace NaNs with the median of the row, required to balance the number of votes per PVS.
    nanvotes = np.where(np.isnan(votes))
    row_mean = np.nanmedian(votes, axis=1)
    votes[nanvotes] = np.take(row_mean, nanvotes[0])

    # create a temporary dataframe to store the scores and the PVS names from the same SRC.
    df = pd.DataFrame({'score': votes.flatten(),
                       'group': np.repeat(pvsnames, repeats=nb_obs)})

    # call the pairwise Tukey test and store it in a dataframe named intraSRCsignf.
    tukeyout = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
    tukey_results = pd.DataFrame(data=tukeyout._results_table.data[1:], columns=tukeyout._results_table.data[0],
                                 index=None)
    intraSRCsignf = pd.concat([intraSRCsignf, tukey_results], ignore_index=True)
    c += 1

# add a column to the dataframe to store the statistical significance of the pairwise Tukey test
# -1 means group1 is significantly better than group2
# 0 means there is no significant difference between the groups.
# 1 means group2 is significantly better than group1

for i in range(intraSRCsignf.shape[0]):
    if intraSRCsignf.loc[i, 'reject']:
        if intraSRCsignf.loc[i, 'meandiff'] < 0:
            intraSRCsignf.loc[i, 'signf'] = 1
        else:
            intraSRCsignf.loc[i, 'signf'] = -1
    else:
        intraSRCsignf.loc[i, 'signf'] = 0

intraSRCsignf.drop(['meandiff', 'p-adj', 'lower', 'upper', 'reject'], axis=1, inplace=True)
# save the results to a csv file to be used in the evaluation script.
intraSRCsignf.to_csv('data/intraSRC_signf.csv')
