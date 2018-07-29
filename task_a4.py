'''
run this module as a script to conduct benchmarking experiments
also performs hypothesis tests to find the best model

note:
    1. regressors and classifiers are compared separately
    2. all estimators that need hyper-parameter tunigh are wrapped
       with GridSearchCV (with refit=True) and the whole strategy is
       validated in the outer loop
    3. more details in the report
'''
import os

# data manipulation utilities
import pickle
import pandas as pd
from tabulate import tabulate

# numeric computations
import numpy as np

# sklearn utilities and estimators
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

# stat utils
from scipy.stats import norm

# reusing code from other tasks
import task_a2


def get_estimators(variety):
    '''
    all estimators under consideration are defined here
    returns regressors or classifiers as per request

    variety argument takes "regressor" or "classifier" as values
    '''
    # check validity of input argument
    if variety != "regressor" and variety != "classifier":
        raise ValueError("Unrecognized variety of estimator")

    # initialize list of regressors
    r_est = []  # regressors
    c_est = []  # classifiers

    # Dummy Regressor
    combo = DummyRegressor()
    r_est.append((combo, "DR"))

    # Linear Regressor
    param_grid = {
        "fit_intercept": [True, False]
        }
    combo = GridSearchCV(LinearRegression(),
                         param_grid,
                         refit=True,
                         scoring="neg_mean_squared_error",
                         n_jobs=-1)
    r_est.append((combo, "LR"))

    # SVR
    param_grid = {
        "C": [0.1, 1, 10],
        }
    # uses RBF kernel by default
    combo = GridSearchCV(SVR(),
                         param_grid,
                         refit=True,
                         scoring="neg_mean_squared_error",
                         n_jobs=-1)
    r_est.append((combo, "SVR"))

    # neural_network Regressor
    param_grid = {
        "hidden_layer_sizes": [(32, 32,), (16, 16, 16,)],
        "learning_rate_init": [0.01, 0.001]
        }
    combo = GridSearchCV(MLPRegressor(),
                         param_grid,
                         refit=True,
                         scoring="neg_mean_squared_error",
                         n_jobs=-1)
    r_est.append((combo, "NNR"))

    # Ensemble Regressor
    combo = BaggingRegressor(base_estimator=None,  # uses trees by default
                             n_estimators=20,  # atleast 10 as per instructions
                             n_jobs=-1)
    r_est.append((combo, "ER"))

    # Dummy Classifier
    combo = DummyClassifier()
    c_est.append((combo, "DC"))

    # Logistic Classifier
    param_grid = {
        "fit_intercept": [True, False],
        "C": [0.1, 1, 10]
        }
    combo = GridSearchCV(LogisticRegression(),
                         param_grid,
                         refit=True,
                         n_jobs=-1)
    c_est.append((combo, "LC"))

    # SVC
    param_grid = {
        "C": [0.1, 1, 10]
        }
    # uses RBF kernel by default
    combo = GridSearchCV(SVC(),
                         param_grid,
                         refit=True,
                         n_jobs=-1)
    c_est.append((combo, "SVC"))

    # neural_network classifier
    param_grid = {
        "hidden_layer_sizes": [(32, 32,), (16, 16, 16,)],
        "learning_rate_init": [0.01, 0.001]
        }
    combo = GridSearchCV(MLPClassifier(),
                         param_grid,
                         refit=True,
                         n_jobs=-1)
    c_est.append((combo, "NNC"))

    # Ensemble Classifier
    combo = BaggingClassifier(base_estimator=None,  # uses trees by default
                              n_estimators=20,  # atleast 10 as in instructions
                              n_jobs=-1)
    c_est.append((combo, "EC"))

    # group estimators and return as per variety
    if variety == "regressor":
        return r_est
    return c_est


def cond_mean(truth, preds):
    '''
    custom scorer
    gives mean of squared error conditioned on training split
    '''
    sq_diff = (truth - preds)**2
    return np.mean(sq_diff)


def cond_var(truth, preds):
    '''
    custom scorer
    gives variance of MSE conditioned on training split
    '''
    sq_diff = (truth - preds)**2
    return (np.var(sq_diff, ddof=1))/truth.shape[0]


def cond_01_mean(truth, preds):
    '''
    custom scorer
    gives mean of squared error conditioned on training split
    '''
    loss = (truth != preds).astype(float)
    return np.mean(loss)


def cond_01_var(truth, preds):
    '''
    custom scorer
    gives variance of MSE conditioned on training split
    '''
    loss = (truth != preds).astype(float)
    return (np.var(loss, ddof=1))/truth.shape[0]


def get_estimator_performance(e_dat_list, score_dict, cv_split):
    '''
    e_dat: dict of estimator and data
    score_dict: scores for cval
    cv_split: consistent split for cval

    returns the score as tuples for all e_dat
    '''
    rval = []

    for e_dat in e_dat_list:
        print("Training: "+e_dat["name"])
        buff = {}
        score = cross_validate(e_dat["estimator"],
                               e_dat["x data"],
                               e_dat["y data"],
                               scoring=score_dict,
                               return_train_score=False,
                               cv=cv_split)
        # fill the results buffer
        buff["name"] = e_dat["name"]
        # the means are still estimates conditional on
        # training set (but they have low variance)
        buff["score"] = [np.mean(score["test_"+str(k)]) for k in score_dict]
        rval.append(buff)
    return rval


def save_score(fname):
    '''
    the main function:
    speaks for itself
    '''

    print("Importing Data...")
    wdf = task_a2.get_data("winequality")

    # log transform
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
    lwdf = wdf.copy()
    # put a shift (0.5*(leastcount=10e-2)) on citric_acid
    lwdf["citric acid"] = lwdf["citric acid"]+10e-2*0.5
    lwdf[transform_list] = np.log(lwdf[transform_list])
    wdf = lwdf

    print("Processing Data...")
    # shuffle data
    wdf = shuffle(wdf)
    # get dummy variables
    wdf = pd.get_dummies(wdf)
    # reuse the same CV split for all models
    cv_split = KFold(n_splits=5)
    # separate features and targets
    y_data = wdf["quality"]
    x_data = wdf.iloc[:, 1:]

    # for both regression and classification
    for kind in ["classifier", "regressor"]:
        print("Seting up the necessary "+kind+"s...")
        ests = get_estimators(kind)

        # create a dictionary of scorers
        if kind == "regressor":
            score_dict = {
                "mean": metrics.make_scorer(cond_mean),
                "var": metrics.make_scorer(cond_var)
                }
        else:
            score_dict = {
                "mean01": metrics.make_scorer(cond_01_mean),
                "var01": metrics.make_scorer(cond_01_var)
                }
        # column masks for features
        fea_masks = ["", "CHM", "COL"]

        # list to hold estimators and corresponding data
        test_list = []
        for est in ests:
            for mask in fea_masks:
                buff = {}
                buff["name"] = est[1]+" "+mask
                buff["estimator"] = est[0]
                if mask == "":  # all features
                    buff["x data"] = x_data
                elif mask == "CHM":  # only chemical composition
                    buff["x data"] = x_data.iloc[:, 0:-2]
                elif mask == "COL":  # only color
                    buff["x data"] = x_data.iloc[:, -2:]
                else:
                    raise ValueError("Unknown mask specification")
                buff["y data"] = y_data
                test_list.append(buff)

        score = get_estimator_performance(test_list,
                                          score_dict,
                                          cv_split)
        # save scores to file
        with open(kind+"."+fname, 'wb') as handle:
            pickle.dump(score, handle)


def tabulate_results(fnames):
    '''
    reads the files and tabulates results
    this is not a reusable function
    many stuffs are hard coded
    '''
    for fname in fnames:
        if fname == "classifier.score":
            print("Comparison of Regressors\n")
            # prep for 95% CI calculation
            alpha = 0.05
            z_val = norm.ppf(alpha/2)
            # read scores from file
            with open(fname, 'rb') as handle:
                score = pickle.load(handle)
            table = []
            for dic in score:
                table.append(
                    [dic["name"],
                     *dic["score"],
                     "(" +
                     f"{dic['score'][0]+z_val*np.sqrt(dic['score'][1]):.4f}" +
                     ", " +
                     f"{dic['score'][0]-z_val*np.sqrt(dic['score'][1]):.4f}" +
                     ")"])
            print(tabulate(table,
                           headers=["Classifier",
                                    "Mean(0/1 loss)",
                                    "Var(Mean(0/1 loss))",
                                    "95% Confidence Interval"],
                           tablefmt="latex",
                           floatfmt=".6f"))
            print("\n\n")
        if fname == "regressor.score":
            print("Comparison of Regressors\n")
            # prep for 95% CI calculation
            alpha = 0.05
            z_val = norm.ppf(alpha/2)
            # read scores from file
            with open(fname, 'rb') as handle:
                score = pickle.load(handle)
            table = []
            for dic in score:
                table.append(
                    [dic["name"],
                     *dic["score"],
                     "(" +
                     f"{dic['score'][0]+z_val*np.sqrt(dic['score'][1]):.4f}" +
                     ", " +
                     f"{dic['score'][0]-z_val*np.sqrt(dic['score'][1]):.4f}" +
                     ")"])
            print(tabulate(table,
                           headers=["Regressor",
                                    "Mean(SE)",
                                    "Variance(MSE)",
                                    "95% Confidence Interval"],
                           tablefmt="latex",
                           floatfmt=".6f"))
            print("\n\n")


def get_p_table(fname):
    '''
    this function is only for regressor scores
    it does the following
    1. reads the given file
    2. creates a table with p-value comparing every model
       with every other model
    3. p values are calculated using a std.normal as null distribution
       and by taking 1-cdf(diff.means/sqrt(sum(var_means)))
    '''
    if fname == "regressor.score":
        print("\n\nComparing Regressors")
    elif fname == "classifier.score":
        print("\n\nComparing Clasifiers")
    else:
        raise ValueError("cant process this file")

    # hypothesis testing parameters
    alpha = 0.05
    print("Testing Hypotheses with alpha = "+str(alpha))
    print("Alternative hypothesis is loss of row index < loss of column index")

    with open(fname, 'rb') as handle:
        score = pickle.load(handle)

    # initialize the result list
    result1 = []
    result2 = []
    header = [" "]

    for dic1 in score:
        # initialize the outer loop buffer
        out_buff1 = []
        out_buff2 = []
        out_buff1.append(dic1["name"])
        out_buff2.append(dic1["name"])
        for dic2 in score:
            # find p value
            h_mean = dic2["score"][0] - dic1["score"][0]  # subtract mean
            h_sd = np.sqrt(dic1["score"][1] + dic2["score"][1])  # add var
            p_val = 1-norm.cdf(h_mean/h_sd)  # as mean as per H0 is 0
            # test hypothesis
            if p_val < alpha:
                h_result = "T"
            else:
                h_result = "-"
            in_buff1 = f"{p_val:.2f}"
            in_buff2 = h_result
            # add result to outer buffer
            out_buff1.append(in_buff1)
            out_buff2.append(in_buff2)
        # add out buffer to result
        result1.append(out_buff1)
        result2.append(out_buff2)

        # fill the header
        header.append(dic1["name"])

    print(tabulate(result1, headers=header, tablefmt="latex"))
    print(tabulate(result2, headers=header, tablefmt="latex"))


if __name__ == "__main__":
    if not (os.path.exists("classifier.score") and
            os.path.exists("regressor.score")):
        save_score("score")

    # report results
    tabulate_results(["classifier.score", "regressor.score"])

    # perform hypothesis testing
    get_p_table("regressor.score")
    get_p_table("classifier.score")
