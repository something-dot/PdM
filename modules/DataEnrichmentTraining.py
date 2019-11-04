# Import system libraries
import sys
# Import libraries for data manipulation and analysis
import pandas as pd
import numpy as np
# Import libraries for imputation
from fancyimpute import KNN
from fancyimpute import IterativeImputer
# Import libraries from sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF


def profile(func):
    import time
    import logging

    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        logging.info(time.time() - started_at)
        return result
    return wrap


# Creates missing values in data
def createMissingValues(df, target='y', percentNan=0.1) -> pd.core.frame.DataFrame:
    copyDf = df.copy(deep=True)
    # tore target columnn
    targetColumn = copyDf.loc[:,target]
    # Store index of target column
    targetIdx = copyDf.columns.get_loc(target)
    # Subset data to not include the target
    copyDf = copyDf.loc[:, df.columns != target]
    # Create data with null values based on some percentage
    dataWithNan = copyDf.mask(np.random.random(copyDf.shape) < percentNan)
    # Recreate dataframe with target column using destructive technique (no need for assignment)
    dataWithNan.insert(loc=targetIdx,column=target, value=targetColumn)
    return dataWithNan

# Imputing technique
@profile
def mice(df, target='y') -> pd.core.frame.DataFrame:
    copyDf = df.copy(deep=True)
    # Store target column
    targetColumn = copyDf.loc[:, target]
    # Store index of target column
    targetIdx = copyDf.columns.get_loc(target)
    # Subset data to not include the target
    copyDf = copyDf.loc[:, df.columns != target]
    # Initialize technique and transform data
    miceImputer = IterativeImputer()
    imputedData = pd.DataFrame(miceImputer.fit_transform(copyDf))
    # Recreate dataframe with target column using destructive technique (no need for assignment)
    imputedData.insert(loc=targetIdx, column=target, value=targetColumn)
    return imputedData

#Imputing technique
@profile
def exponentialMovingAverage(df, alphaVals, adjust=False, ignore_na=True, target='y', fillNull=0) -> pd.core.frame.DataFrame:
    copyDf = df.copy(deep=True)
    # Store target columnn
    targetColumn = copyDf.loc[:, target]
    # Store index of target column
    targetIdx = copyDf.columns.get_loc(target)
    # Subset data to not include the target
    copyDf = copyDf.loc[:, df.columns != target]
    # Apply exponential moving average with specific alpha value
    copyDf[copyDf.isna()] = copyDf.ewm(alpha=alphaVals, adjust=adjust, ignore_na=ignore_na).mean().fillna(fillNull)
    # Recreate dataframe with target column using destructive technique (no need for assignment)
    copyDf.insert(loc=targetIdx, column=target, value=targetColumn)
    return copyDf

# Imputing technique
@profile
def knnImputation(df, kneighbors=3, target='y') -> pd.core.frame.DataFrame:
    copyDf = df.copy(deep=True)
    # Store target column
    targetColumn = copyDf.loc[:, target]
    # Store index of target column
    targetIdx = copyDf.columns.get_loc(target)
    # Subset data to not include the target
    copyDf = copyDf.loc[:, df.columns != target]
    # Initialize technique and transform data
    knnImputer = KNN(kneighbors)
    knnData = pd.DataFrame(knnImputer.fit_transform(copyDf))
    # Recreate dataframe with target column using destructive technique (no need for assignment)
    knnData.insert(loc=targetIdx, column=target, value=targetColumn)
    return knnData

# Helper function for calculating mean absolute error per feature
def meanAbsoluteError(true,pred) -> list:
    # True and pred should be same shape and should be dataframe
    n,m = true.shape
    # Convert to 2-by-2 array
    arrayOne = true.values
    arrayTwo = pred.values
    # Calculate mape along the features of each array
    result = [mean_absolute_error(arrayOne[:,columnIdx], arrayTwo[:,columnIdx]) * 100 for columnIdx in range(m)]
    return result


def bestEnrichmentTechnique(missingDf, realDf, target='y', alphas=np.linspace(0.1, 1, 10), knns=list(range(3,7,1))) -> str:
    # Create empty dataframe to store results of data enrichment technique
    testResults = pd.DataFrame()
    # Create a features column storing all the column names
    testResults["features"] = realDf.columns.tolist()
    # Mice
    micedDf = mice(missingDf, target=target)
    # Calculate mean absolute percent error for mice computation
    testResults["mice"] = meanAbsoluteError(realDf, micedDf)
    # Hypertuning Exponential Moving Average
    for alpha in alphas:
        missingDfPerAlpha = createMissingValues(realDf, target='y',percentNan=0.1)
        expDf = exponentialMovingAverage(missingDfPerAlpha, alphaVals=alpha, target=target, fillNull=0)
        # Calculate mean absolute percent error for each alpha
        testResults["mape_exp_" + str(alpha)] = meanAbsoluteError(realDf, expDf)
    # Hypertuning KNN imputation
    for knn in knns:
        knnDf = knnImputation(missingDf, kneighbors=knn, target=target)
        # Calculate mean absolute percent error for each neighbor
        testResults["mape_knn_" + str(knn)] = meanAbsoluteError(realDf, knnDf)
    # #Matrix Factorization imputation
    # factorizationDf = matrixFactorization(missingDf)
    # #Calculate mean absolute percent error for matrix factorization
    # testResults['matrixFactorization'] = meanAbsoluteError(realDf, factorizationDf)
    #Show results for the different techniques and aggregate across all the features
    print(testResults.describe().loc['mean',:])
    return (testResults.describe().loc['mean',:]).idxmin()


def main():
    '''This approach does not consider train/test for imputation & only works because original data has no missing values'''
    # Read in data
    df = pd.read_csv(sys.argv[1])
    # Remove time column, and the categorical columns
    df = df.drop(['time', 'x28', 'x61'], axis=1)
    # Reduce dataset size for local implementation
    # df = df.iloc[:300,:]
    # Create missing values
    missingDf = createMissingValues(df, target='y', percentNan=0.1)
    # String containing best technique and any parameters
    technique = bestEnrichmentTechnique(missingDf, df)
    print(technique)

    # Output new DataFrame into output section
    df.to_csv("../output/imputed_df.csv")

    return

if __name__ == '__main__':
    main()
