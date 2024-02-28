import numpy as np
import pandas as pd
from tools_vmaf_Krasula_Method import Krasula_metric_performance

# This file uses the vmaf library to calculate the performance metrics. Since the whole vmaf library is excessive for
# this task, we include the necessary functions in the tools_vmaf_Krasula_Method.py file


def get_metric_score_diff(metric_preds, signf_tr):
    '''
    This function calculates the difference between the metric scores of two stimuli based on the ground truth pairs.
    :param metric_preds: csv file containing the metric scores for each stimulus
    :param signf_tr: csv file containing the ground truth pairs
    :return: returns the difference between the metric scores of two stimuli for each pair
    '''
    # create an empty dataframe to store metric score differences
    msd_cols = ['group1', 'group2', 'pred_difference']
    metric_score_diffs = pd.DataFrame(columns=msd_cols)
    for i in range(signf_tr.shape[0]):
        metric_score_diffs.loc[i, 'group1'] = signf_tr.loc[i, 'group1']
        metric_score_diffs.loc[i, 'group2'] = signf_tr.loc[i, 'group2']
        pvs1_scores = metric_preds.loc[metric_preds['ppc'] == metric_score_diffs.loc[i, 'group1']]
        pvs2_scores = metric_preds.loc[metric_preds['ppc'] == metric_score_diffs.loc[i, 'group2']]
        metric_score_diffs.iloc[i, -1] = pvs1_scores.iloc[0, 1] - pvs2_scores.iloc[0, 1]
    return metric_score_diffs


def get_metric_performance(metric_predictions, ground_truth_signficance):
    """
    This function calculates the performance of the metric based on the Krasula's method.
    Check the corresponding papers for more information.
    :param metric_predictions: csv file containing the metric scores for each stimulus.
    :param ground_truth_signficance: csv file containing the ground truth pairs and statistical significance of their differences.
    :return: returns metric performances in terms of Diff_Sim_AUC and Better_Worse_CC.
    """
    # calls the function above to calculate the metric score differences
    prediction_differences = get_metric_score_diff(metric_predictions, ground_truth_signficance)
    obj_score_diff = np.expand_dims(prediction_differences['pred_difference'].to_numpy().T, axis=0)
    gt_signf = np.expand_dims(ground_truth_signficance['signf'].to_numpy().T, axis=0)

    # calls the Krasula's method to calculate the performance metrics
    results = Krasula_metric_performance(obj_score_diff, gt_signf)
    Diff_Sim_AUC = results['AUC_DS'][0]
    Better_Worse_CC = results['CC_0'][0]

    return Diff_Sim_AUC, Better_Worse_CC, prediction_differences


ground_truth = pd.read_csv('data/intraSRC_signf.csv')  # ground truth pairs and their statistical significance
metric_predictions = pd.read_csv('data/pcqm_predictions.csv')  # metric scores for each stimulus

# The script expects the metric scores to be in a way that higher scores indicate better quality.
# If the metric scores are in the opposite direction, the script will return a negative value for Better_Worse_CC.
# In that case, you can multiply the metric scores by -1 and try again. It will not affect the Diff_Sim_AUC.
# As an example, PCQM returns lower scores for better quality, so we multiply the metric scores by -1.
metric_predictions['predictions'] *= -1

Diff_Sim_AUC, Better_Worse_CC, metric_pred_diffs = get_metric_performance(metric_predictions, ground_truth)
print('Diff_Sim_AUC: ', Diff_Sim_AUC)
print('Better_Worse_CC: ', Better_Worse_CC)
metric_pred_diffs.to_csv('data/pcqm_prediction_differences_testset.csv')





