#!/usr/bin/env python3
__author__ = 'Jonathan Nagel'
__email__ = 'jon@jonnagel.com'
from sklearn import cross_validation, feature_extraction, grid_search, feature_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import pickle
import re

def load_data_train(file_name):
    ''' load and clean sf data'''

    float_cols = [0, 29]
    int_cols = [3, 4, 5, 6, 12, 20, 21, 23, 25, 26, 27, 28, 30]
    date_cols = [14, 22]
    drop_cols = ['X2', 'X3']
    # what is up with col 11?

    def to_numbs(s):

        if re.findall(r'<', s):
            return(0)
        if re.findall(r'\+', s):
            return(11)
        if re.findall(r'\d+', s):
            return(int(re.findall(r'\d+', s)[0]))
        else:
            return(None)
    chars_to_rm = '$%,'
    rgx = re.compile('[%s]' % chars_to_rm)
    # stop whining on import
    dt = {15: pd.np.str}
    df = pd.read_csv(file_name, dtype=dt)
    # a choice to dump incomplete cases
    df = df.dropna()
    # if it's an object remove chars
    for ob in [x for x in df.columns if df[x].dtype == 'O']:
        df[ob] = df[ob].apply(lambda x: rgx.sub('', str(x)))
    # words... we don't need them.
    df.X7 = [int(str(x)[1:3]) for x in df.X7]
    # fix silliness with col
    df.ix[:, 10] = [to_numbs(x) for x in df.X11]
    # fix dtypes
    for c in date_cols:
        df.ix[:, c] = [pd.datetime.strptime(x, '%b-%y') for x in df.ix[:, c]]
    for c in float_cols:
        df.ix[:, c] = df.ix[:, c].astype('float')
    for c in int_cols:
        df.ix[:, c] = df.ix[:, c].astype('int64')
    return df

def load_data_test(file_name):
    ''' preen test data'''

    int_cols = [4, 5]
    chars_to_rm = '$%,'
    rgx = re.compile('[%s]' % chars_to_rm)
    dt = {15: pd.np.str}
    df = pd.read_csv(file_name, dtype=dt)
    for ob in [x for x in df.columns if df[x].dtype == 'O']:
        df[ob] = df[ob].apply(lambda x: rgx.sub('', str(x)))
    for c in int_cols:
        df.ix[:, c] = df.ix[:, c].astype('int64')
    return df.ix[:, ['X4', 'X6', 'X8', 'X9']].dropna()

def get_best_cols(df):
    ''' select best cols with RFE '''

    # factors
    cols_to_factor = [pd.get_dummies(df.X7),
                      pd.get_dummies(df.X8),
                      pd.get_dummies(df.X9),
                      pd.get_dummies(df.X11),
                      pd.get_dummies(df.X12),
                      pd.get_dummies(df.X14),
                      pd.get_dummies(df.X12),
                      pd.get_dummies(df.X14),
                      pd.get_dummies(df.X32)]
    # dataframe with factors blown out
    df_f = pd.concat(cols_to_factor, axis=1)
    # numerics
    RFE_col_list = ['X4', 'X5', 'X6', 'X13',
                    'X21', 'X22', 'X29', 'X30',
                    'X31']
    # dataframe with numerics
    df_n = df.ix[:, RFE_col_list]
    X = np.asarray(df_n)
    X = StandardScaler().fit_transform(X)
    # add in factors
    X = np.concatenate([X, np.asarray(df_f)], axis=1)
    # leave y alone
    y = df.X1
    # I don't like to guess yes this is only linear relationships
    estimator = SVR(kernel="lineary")
    selector = RFE(estimator, 40, step=2)
    selector = selector.fit(X, y)
    # make index for merged df, yes this whines
    df_index = df_n.columns + df_f.columns
    best_cols = df_index[selector.support_]
    return best_cols

def pfp(d_f):
    ''' prep for pipes '''

    df_trimmed = d_f.ix[:, ['X4', 'X6', 'X8', 'X9']]
    df_trimmed_f = pd.get_dummies(df_trimmed.ix[:, ['X8', 'X9']])
    df_trimmed_n = df_trimmed.ix[:, ['X4', 'X6']]
    # whining because ints
    df_trimmed_n = df_trimmed_n.apply(lambda x: StandardScaler().fit_transform(x))
    df_trimmed_b = pd.concat([df_trimmed_n, df_trimmed_f], axis=1)
    return df_trimmed_b

# load & clean sf data
df = load_data_train('Data for Cleaning & Modeling.csv')
df_test = load_data_test('Holdout for Testing.csv')
# what cols?
get_best_cols(df)
# most support for X4, X6, X8, X9

# use while developing model
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_trimmed_b,
#                                                     df.ix[:,0],
#                                                     test_size=0.2,
#                                                     random_state=42)

# break into matrices
X_train, y_train = pfp(df), df.ix[:, 0]
X_test, y_test = pfp(df_test), None

##
# random forest
##
# piperfr = Pipeline([
#     ("rfr", RandomForestRegressor())
# ])
# paramsrfr = dict(
#     # rfr__max_features=['auto', 'sqrt', 'log2'],
#     rfr__max_leaf_nodes=list(range(2, 50)),
#     # rfr__min_samples_leaf=list(range(1,10)),
#     # rfr__min_samples_split=list(range(1,10)),
#     # rfr__min_weight_fraction_leaf=[0, 0.5],
#     rfr__n_estimators=list(range(13, 23)),
#     # rfr__oob_score=[True, False],
#     rfr__bootstrap=[True, False],
# )
# g_s_rfr = grid_search.GridSearchCV(piperfr, paramsrfr, n_jobs=4)
# g_s_rfr.fit(X_train, y_train)
#
# best_rfr = g_s_rfr.best_estimator_
# pickle.dump(best_rfr, open("best_rfr.p", "wb"))

best_rfr = pickle.load(open("best_rfr.p", "rb"))
best_rfr.fit(X_train, y_train)
preds_rfr = pd.DataFrame(best_rfr.predict(X_test))

##
# logit
##
# def rmsejn(preds, actual):
#     return np.sqrt(np.mean((actual - preds)**2))
# # monkey patch around sklearn shortcomings
# class local_LR(LogisticRegression):
#     def score(self, X, y):
#         try:
#             # flipped because it's a measure of error
#             s = 1-rmsejn(self.predict(X_test), y_test)
#         except:
#             s = 0
#             pass
#         return s
# # pipe for logit
# pipelr = Pipeline([
#     ("lr", local_LR(penalty='l2'))
# ])
# # paramater dict for logit
# paramslr = dict(
#     lr__fit_intercept = [True, False],
#     lr__C = [1e3, 1e4, 1e5, 1e6, 1e7],
# )
# g_s_lr = grid_search.GridSearchCV(pipelr, paramslr, n_jobs=4)
# g_s_lr.fit(X_train, y_train)
#
# best_lr = g_s_lr.best_estimator_
# pickle.dump(best_lr, open("best_lr.p", "wb"))

best_lr = pickle.load(open("best_lr.p", "rb"))
best_lr.fit(X_train, y_train)
preds_lr = pd.DataFrame(best_lr.predict(X_test))

# combine and output as per instructions
preds_both = pd.concat([preds_rfr, preds_lr], axis=1)
preds_both.columns = ['rfr', 'lr']
pd.DataFrame(preds_both).to_csv('ResultsfromJonathanNagel.csv', index=False)
