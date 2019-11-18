#sklearn packages
from sklearn.model_selection import  GridSearchCV
#Import libraries for data manipulation and analysis
import pandas as pd
import numpy as np

class EstimatorSelectionHelper:
    '''
    This class allows for hypertuning of multiple models and parameters at once.

    Inputs:
    models       A dictionary containing keys of models names and values consisting of instantiated sklearn models
    params       A dictionary containing either matching keys to the models or no params at all

    Output
    df           A dataframe consisting of scores for the tuning of multiple models

    i.e.        models = {'Logistic Regression': LogisticRegression()}
                params = {'Logistic Regression':  {'C':[0.1,1.0,10], 'penalty':['l1','l2']}}
                helper = EstimatorSelectionHelper(models, params)
                helper.fit(X,y)
                helper.score_summary()

    '''

    # class constructor
    def __init__(self, models, params):
        # checking consistency of estimator and paramater keys
        if len(models.keys()) < len(params.keys()):
            raise ValueError("Need more estimators.")
        missing_estimators = list(set(params.keys()) - set(models.keys()))
        if len(missing_estimators) > 0:
            raise ValueError("Following parameters: {} are missing estimators.".format(missing_estimators))
        # instance variables
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.cv_searches = {}

    # fit method which can be modified
    def fit(self, X, y, cv=5, n_jobs=3, verbose=1, scoring=None, refit=False, return_train_score=True):
        # looping throught the keys
        for key in self.keys:
            model = self.models[key]
            # running gridsearch cv with parameter tuning
            if key in self.params:
                print("Running GridSearchCV for {}.".format(key))
                params = self.params[key]
                gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                                  verbose=verbose, scoring=scoring, refit=refit,
                                  return_train_score=return_train_score)
                gs.fit(X, y)
                # storing grid searches in dictionary
                self.grid_searches[key] = gs
            # running cv without parameter tuning
            else:
                print("Running CV for {}.".format(key))
                cv = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs,
                                     verbose=verbose, scoring=scoring)
                self.cv_searches[key] = cv

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        #         rows = []
        #         for k in self.cv_searches:
        #             print(k)
        #             r = cv[k]['test_score']
        #             p = [np.nan] * (len(columns) - 5)
        #             df = df.append(row(k,r,p), ignore_index=True)

        return df[columns]