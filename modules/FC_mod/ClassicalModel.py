# Import system libraries
import sys
import math
# Import libraries for data manipulation and analysis
import pandas as pd
import numpy as np
# Importing custom classes
from Modules.Classes.ClassicalModel import EstimatorSelectionHelper
from Modules.Classes.TimeShift import curveShift
# Import sklearn process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import sklearn methods
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def splitIntoXAndY(data, target):
    copyDf = data.copy(deep=True)
    # Subset train data to not include the target
    copyDf_x = copyDf.loc[:, copyDf.columns != target]
    # Keep target values separately
    copyDf_y = copyDf[target]
    return copyDf_x, copyDf_y


def main():
    # Assign params
    dataPath, DATA_SPLIT_PCT, RANDOM_STATE = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    shift = -25 # automate this
    target = 'y' # automate this
    # Read in data
    df = pd.read_csv(dataPath)
    # Remove time column, and the categorical columns
    df = df.drop(['time', 'x28', 'x61'], axis=1) #automate dropping such columns
    # Reduce dataset size for local implementation
    df = df.iloc[:300,:]
    # Curve shift the data
    df = curveShift(df, shift_by=shift)
    df = df.shift(target).show()
    # Train & test split
    df_train, df_valid = train_test_split(df, test_size=DATA_SPLIT_PCT, random_state=RANDOM_STATE)
    # Subset train data to not include the target and keep target values separately
    df_train_x, df_train_y = splitIntoXAndY(df_train, target)
    # Subset valid data to not include the target and keep target values separately
    df_valid_x, df_valid_y = splitIntoXAndY(df_valid, target)
    # Scaling data
    scaler = StandardScaler().fit(df_train_x)
    df_train_x_rescaled = scaler.transform(df_train_x)
    df_valid_x_rescaled = scaler.transform(df_valid_x)
    # Classical models to use in the class #automate this
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier()
    }
    # Classical model parameters to tune #automate this
    params = {
        'LogisticRegression': {'C': [0.01, 1, 100]},
        'RandomForestClassifier': {'n_estimators': [10, 50, 100]},
        'AdaBoostClassifier': {'n_estimators': [10, 50, 100]},
        'SVC': [
            {'kernel': ['linear'], 'C': [1, 10]},
            {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
        ],
        'KNN': {'n_neighbors': [3]}
    }
    # Instantiating class which is internally running cross validation
    helper = EstimatorSelectionHelper(models, params)
    # Fitting on rescaled training data
    helper.fit(df_train_x_rescaled, df_train_y, scoring='roc_auc', n_jobs=3, verbose=True)
    # Obtaining the summary & determining best model
    resultsDf = helper.score_summary(sort_by='max_score')
    #Storing the max result paramaters and not retaining any values that are None
    bestParamDictionary = resultsDf.iloc[0,5:].to_dict()
    bestParamDictionary = {key:bestParamDictionary[key] for key in bestParamDictionary.keys() if not math.isnan(bestParamDictionary[key])}
    #Obtaining the string for best method, instantiating it, and setting the parameters
    bestMethodString = resultsDf['estimator'].iloc[0]
    bestMethod = models[bestMethodString]
    bestMethod.set_params(**bestParamDictionary)
    #Performing on valid data and obtaining predictions
    bestMethod.fit(df_train_x_rescaled, df_train_y)
    yPred = bestMethod.predict(df_valid_x_rescaled)
    yTrue = df_valid_y
    return yPred, yTrue

if __name__ == '__main__':
    print(main())


