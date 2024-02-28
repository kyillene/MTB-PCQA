import numpy as np
import scipy

"""
The functions below are taken from the VMAF repository and are used to run the evaluation methodology based on 
Krasula's method. For more information on the evaluation methodology, please refer to the following paper: 

L. Krasula, K. Fliegel, P. Le Callet and M. Klíma, "On the accuracy of objective image and video quality models: New 
methodology for performance evaluation," 2016 Eighth International Conference on Quality of Multimedia Experience (
QoMEX), Lisbon, Portugal, 2016, pp. 1-6, doi: 10.1109/QoMEX.2016.7498936.

VMAF Repository: https://github.com/Netflix/vmaf
"""


def empty_object():
    return type('', (), {})()


def midrank(x):

    J, Z = zip(*sorted(enumerate(x), key=lambda x:x[1]))
    J = list(J)
    Z = list(Z)
    Z.append(Z[-1]+1)
    N = len(x)
    T = np.zeros(N)

    i = 1
    while i <= N:
        a = i
        j = a
        while Z[j-1] == Z[a-1]:
            j = j + 1
        b = j - 1
        for k in range(a, b+1):
            T[k-1] = (a + b) / 2
        i = b + 1

    # T(J)=T;
    T2 = np.zeros(N)
    T2[J] = T

    return T2

def fastDeLong(samples):
    # %FASTDELONGCOV
    # %The fast version of DeLong's method for computing the covariance of
    # %unadjusted AUC.
    # %% Reference:
    # % @article{sun2014fast,
    # %   title={Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
    # %   author={Xu Sun and Weichao Xu},
    # %   journal={IEEE Signal Processing Letters},
    # %   volume={21},
    # %   number={11},
    # %   pages={1389--1393},
    # %   year={2014},
    # %   publisher={IEEE}
    # % }
    # %% [aucs, delongcov] = fastDeLong(samples)
    # %%
    # % Edited by Xu Sun.
    # % Homepage: https://pamixsun.github.io
    # % Version: 2014/12
    # %%
    if np.sum(samples.spsizes) != samples.ratings.shape[1] or len(samples.spsizes) != 2:
        assert False, 'Argument mismatch error'

    z = samples.ratings
    m, n = samples.spsizes
    x = z[:, :m]
    y = z[:, m:]
    k = z.shape[0]


    tx = np.zeros([k, m])
    ty = np.zeros([k, n])
    tz = np.zeros([k, m + n])
    for r in range(k):
        tx[r, :] = midrank(x[r, :])
        ty[r, :] = midrank(y[r, :])
        tz[r, :] = midrank(z[r, :])

    aucs = np.sum(tz[:, :m], axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov, v01, v10


def calpvalue(aucs, sigma):
    # function pvalue = calpvalue(aucs, sigma)
    # l = [1, -1];
    # z = abs(diff(aucs)) / sqrt(l * sigma * l');
    # pvalue = 2 * (1 - normcdf(z, 0, 1));
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    pvalue = 2 * (1 - scipy.stats.norm.cdf(z, loc=0, scale=1))
    return pvalue


def indices(a, func):
    """
    Get indices of elements in an array which satisfies func
    >>> indices([1, 2, 3, 4], lambda x: x>2)
    [2, 3]
    >>> indices([1, 2, 3, 4], lambda x: x==2.5)
    []
    >>> indices([1, 2, 3, 4], lambda x: x>1 and x<=3)
    [1, 2]
    >>> indices([1, 2, 3, 4], lambda x: x in [2, 4])
    [1, 3]
    >>> indices([1,2,3,1,2,3,1,2,3], lambda x: x > 2)
    [2, 5, 8]
    """
    return [i for (i, val) in enumerate(a) if func(val)]


def significanceBinomial(p1, p2, N):
    p = (p1 + p2) / 2.0
    sigmaP1P2 = np.sqrt(p * (1.0 - p) * 2.0 / N)
    z = abs(p1 - p2) / sigmaP1P2
    pValue = 2.0 * (1.0 - scipy.stats.norm.cdf(z, 0.0, 1.0))

    return pValue


def Krasula_metric_performance(objScoDif, signif):
    """
    % INPUT:    objScoDif   : differences of objective scores [M,N]
    %                         M : number of metrics
    %                         N : number of pairs
    %           signif      : statistical outcome of paired comparison [1,N]
    %                          0 : no difference
    %                         -1 : first stimulus is worse
    %                          1 : first stimulus is better
    % OUTPUT:   results - structure with following fields
    %
    %           AUC_DS      : Area Under the Curve for Different/Similar ROC
    %                         analysis
    %           pDS_DL      : p-values for AUC_DS from DeLong test
    %           pDS_HM      : p-values for AUC_DS from Hanley and McNeil test
    %           AUC_BW      : Area Under the Curve for Better/Worse ROC
    %                         analysis
    %           pBW_DL      : p-values for AUC_BW from DeLong test
    %           pBW_HM      : p-values for AUC_BW from Hanley and McNeil test
    %           CC_0        : Correct Classification @ DeltaOM = 0 for
    %                         Better/Worse ROC analysis
    %           pCC0_b      : p-values for CC_0 from binomial test
    %           pCC0_F      : p-values for CC_0 from Fisher's exact test
    %           THR         : threshold for 95% probability that the stimuli
    %                         are different
    """
    M = objScoDif.shape[0]
    D = np.abs(objScoDif[:, indices(signif[0], lambda x: x != 0)])
    S = np.abs(objScoDif[:, indices(signif[0], lambda x: x == 0)])
    samples = empty_object()
    samples.spsizes = [D.shape[1], S.shape[1]]
    samples.ratings = np.hstack([D, S])

    # % calculate AUCs
    AUC_DS, C, _, _ = fastDeLong(samples)

    # % significance calculation
    pDS_DL = np.ones([M, M])
    for i in range(1, M):
        for j in range(i + 1, M + 1):
            # http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar
            pDS_DL[i - 1, j - 1] = calpvalue(AUC_DS[[i - 1, j - 1]], C[[[i - 1], [j - 1]], [i - 1, j - 1]])
            pDS_DL[j - 1, i - 1] = pDS_DL[i - 1, j - 1]
    THR = np.percentile(S, 95, axis=1)

    # %%%%%%%%%%%%%%%%%%%%%%% Better / Worse %%%%%%%%%%%%%%%%%%%%%%%%%%%
    B1 = objScoDif[:, indices(signif[0], lambda x: x == 1)]
    B2 = objScoDif[:, indices(signif[0], lambda x: x == -1)]
    B = np.hstack([B1, -B2])
    W = -B
    samples = empty_object()
    samples.ratings = np.hstack([B, W])
    samples.spsizes = [B.shape[1], W.shape[1]]

    # % calculate AUCs

    AUC_BW, C, _, _ = fastDeLong(samples)
    L = B.shape[1] + W.shape[1]
    CC_0 = np.zeros(M)
    for m in range(M):
        CC_0[m] = float(np.sum(B[m, :] > 0) + np.sum(W[m, :] < 0)) / L

    # % significance calculation
    pBW_DL = np.ones([M, M])
    pCC0_b = np.ones([M, M])
    for i in range(1, M):
        for j in range(i + 1, M + 1):
            pBW_DL[i - 1, j - 1] = calpvalue(AUC_BW[[i - 1, j - 1]], C[[[i - 1], [j - 1]], [i - 1, j - 1]])
            pBW_DL[j - 1, i - 1] = pBW_DL[i - 1, j - 1]

            pCC0_b[i - 1, j - 1] = significanceBinomial(CC_0[i - 1], CC_0[j - 1], L)
            pCC0_b[j - 1, i - 1] = pCC0_b[i - 1, j - 1]


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # % Adding outputs to the structure
    result = {
        'AUC_DS': AUC_DS,
        'pDS_DL': pDS_DL,
        'AUC_BW': AUC_BW,
        'pBW_DL': pBW_DL,
        'CC_0': CC_0,
        'pCC0_b': pCC0_b,
        'THR': THR,
    }
    return result