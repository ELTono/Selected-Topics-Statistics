'''
This file contains code for task a5
'''
import numpy as np
from tabulate import tabulate
from scipy.stats import norm
import task_a2


def do_htest(dfa, dfb, label1, label2):
    '''
    a and b are dataframes with same column headings
    performs t-test comparing the two samples separately
    '''
    alpha = 0.05
    headers = ["feature", label1+" p value", label1+" Result", label2+" p value", label2+" Result"]
    result = []

    for feature in dfa.columns[0:-1]:
        x1 = np.mean(dfa[feature])
        x2 = np.mean(dfb[feature])
        v1 = np.var(dfa[feature], ddof=1)/dfa.shape[0]
        v2 = np.var(dfb[feature], ddof=1)/dfa.shape[1]
        # z-statistic
        t = (x1-x2)/np.sqrt(v1+v2)
        # p-value for 1 > 2
        p_val_1g2 = 1 - norm.cdf(t)
        if p_val_1g2 < alpha:
            h_val_1g2 = "T"
        else:
            h_val_1g2 = "-"
        # p-value for 1 < 2
        p_val_2g1 = norm.cdf(t)
        if p_val_2g1 < alpha:
            h_val_2g1 = "T"
        else:
            h_val_2g1 = "-"
        result.append([feature, p_val_1g2, h_val_1g2, p_val_2g1, h_val_2g1])

    print(tabulate(result, headers=headers, floatfmt="0.4f", tablefmt="latex"))

if __name__ == "__main__":
    # get the wine data frame (wdf)
    wdf = task_a2.get_data("winequality")
    # find distinguishing features of good wine
    do_htest(wdf[wdf.quality > 6], wdf[wdf.quality <= 6], label1="G>O:", label2="G<O:")
