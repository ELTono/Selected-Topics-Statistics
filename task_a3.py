'''
This file was actually converted from a notebook
contains code for EDA
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from tabulate import tabulate
import task_a2
import task_a5


def main():
    '''
    main function like in C
    '''
    # toggle for saving (and showing) images
    save = False

    # get the wine data frame (wdf)
    wdf = task_a2.get_data("winequality")

    # Numerical Summaries
    header = [" ", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    print(tabulate(wdf.describe().T, headers=header, tablefmt="latex"))

    # plot histograms
    if save is True:
        plt.figure(figsize=(20, 5))
        filter_list = wdf.color != ""
        for i, w_color in enumerate(["", "red", "white"]):
            if i > 0:
                filter_list = wdf.color == w_color
            plt.subplot(1, 3, i+1)
            plt.hist(wdf[filter_list].quality,
                     bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     align="left")
            plt.title(str(w_color)+" wine quality histogram")
            plt.xlabel("Wine Quality")
            plt.ylabel("Count")
        plt.savefig("histogram.jpg")
        plt.show()

    # Make bar plot
    if save is True:
        i = 1
        fig = plt.figure(figsize=(10, 10))
        for feature in wdf:
            if feature != "color":
                ax = fig.add_subplot(4, 3, i)
                ax.boxplot([wdf[feature][wdf.color == "red"],
                            wdf[feature][wdf.color == "white"]])
                plt.setp(ax, xticklabels=['red', 'white'])
                ax.set_ylabel(feature)
                i += 1
        plt.tight_layout()
        plt.savefig("barplot.jpg")
        plt.show()

    if save is True:
        # matrix scatter plot
        # color coding for wine
        color_dict = {"red":"red", "white":"blue"}
        # getting the color list
        color_list = [color_dict[c] for c in wdf.color]
        # matrix scatter plot
        scatter_matrix(wdf, alpha=0.01, figsize=(20, 30),
                       diagonal='density', color=color_list)
        plt.savefig("mat_scatter.jpg")
        plt.show()

    # log transformation
    transform_list = [
        'fixed acidity',
        'citric acid',
        'volatile acidity',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'sulphates',
        ]
    print("Least count")
    np.min(wdf["citric acid"][wdf["citric acid"] > 0])
    lwdf = wdf.copy()
    # put a shift (0.5*leastcount) on citric_acid (10e-2 is the typical LC)
    lwdf["citric acid"] = lwdf["citric acid"]+10e-2*0.5
    lwdf[transform_list] = np.log(lwdf[transform_list])
    print(tabulate(lwdf.describe().T, headers=header, tablefmt="latex"))

    if save is True:
        # log - matrix scatter plot
        # color coding for wine
        color_dict = {"red": "red", "white": "blue"}
        # getting the color list
        color_list = [color_dict[c] for c in lwdf.color]
        # matrix scatter plot
        scatter_matrix(lwdf, alpha=0.01, figsize=(20, 30), diagonal='density', color=color_list)
        plt.savefig("log_mat_scatter.jpg")
        plt.show()
        
    # rough calculations for quantitative EDA
    from sklearn.metrics import silhouette_score
    print("Red wine mean quality: ", np.mean(wdf["quality"][wdf.color == "red"]))
    print("White wine mean quality: ", np.mean(wdf["quality"][wdf.color == "white"]))
    print("Red wine with quality more than 8: ", len(wdf["quality"][np.logical_and(wdf.quality > 8, wdf.color == "red")]))
    twdf = pd.get_dummies(wdf)
    print("Silhoutte score as per color labels", silhouette_score(twdf.iloc[:,0:-2], twdf.iloc[:,-1]))
    print("Silhoutte score as per random labels", silhouette_score(twdf.iloc[:,0:-2], np.random.randint(0,2, size=(twdf.shape[0]))))

    result = []
    header = [" "]
    for i in lwdf.columns[1:-1]:
        buf = [i]
        header.append(i)
        for j in lwdf.columns[1:-1]:
            buf.append(np.corrcoef(lwdf[i], lwdf[j])[1,0])
        result.append(buf)
    print(tabulate(result, headers=header, tablefmt="latex", floatfmt="0.3f"))

    # comparing red and white wine
    task_a5.do_htest(wdf[wdf.color == "red"], wdf[wdf.color == "white"], label1="R>W:", label2="R<W:")

if __name__ == "__main__":
    main()
