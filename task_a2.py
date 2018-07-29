'''
This file contains code for importing data
'''
# necessary modules
import os
import pandas as pd


def get_data(folder, colors=("white", "red")):
    '''
    locates the following files ( by default) in the folder taken as argument

    winequality-white.csv
    winequality-red.csv

    The data on quality of red-wine and white-wine are aggregated
    into a super data set. This is returned as a pandas dataframe

    Optionally colors can also be given as an argument
    '''
    # initialize pandas df to hold the dataset
    wine_df = pd.DataFrame()

    # read data for each color and append
    for color in colors:
        # get the file paths
        file_path = os.path.join(folder,
                                 "winequality-"+color+".csv")

        # import the CSVs as pandas dataframe
        temp_df = pd.read_csv(file_path, sep=";")

        # add the color variable
        temp_df.loc[:, "color"] = color

        # append to the existing datatrame
        wine_df = wine_df.append(temp_df,
                                 ignore_index=True,
                                 verify_integrity=True)

    # rearrange to make quality the first column
    wine_df = wine_df[["quality"] +
                      [col for col in wine_df if col != "quality"]]

    # return the aggregated dataframe
    return wine_df


if __name__ == "__main__":
    '''
    run testing code if ran as a script
    '''
    WINE_DF = get_data("winequality")

    # check shape
    if WINE_DF.shape == (6497, 13):
        print("The shape of the DF is as expected")
    else:
        print("Shape test FAILED")

    # check red count
    if WINE_DF.color[WINE_DF.color == "red"].count() == 1599:
        print("Red wine count is as expected")
    else:
        print("Red count test FAILED")

    # check white count
    if WINE_DF.color[WINE_DF.color == "white"].count() == 4898:
        print("White wine count is as expected")
    else:
        print("White count test FAILED")
