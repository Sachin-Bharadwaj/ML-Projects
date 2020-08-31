# lbl_xgb_num_feat.py

import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import ensemble
import xgboost as xgb
import itertools

def feature_engineering(df, cat_cols):
    """
    df: pandas dataframe
    cat_cols: list of categorical column names
    """

    combi = list(itertools.combinations(cat_cols, 2))

    for c1,c2 in combi:
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df

def run(fold):

    df = pd.read_csv('../input/train_folds.csv')



    num_features = ["age", "fnlwgt", "education.num", "capital.gain", \
                    "capital.loss", "hours.per.week", "income"]

    cat_cols = [c for c in df.columns if c not in num_features and c not in \
                ["income", "kfolds"]]

    df = feature_engineering(df, cat_cols)

    # find all feature names, all columns are features except id, target, kfolds
    features = [f for f in df.columns if f not in ["income", "kfolds"]]

    # fill all NaNs with NONE
    # Note that converting to string, it does not matter because all are
    # categories in the feature set
    for f in features:
        if f not in num_features: # label encoding only for non-numerical features
            df.loc[:,f] = df[f].astype(str).fillna("NONE")
            # initialize label encoder for this column
            lbl_enc = preprocessing.LabelEncoder()
            # fit the LabelEncoder
            lbl_enc.fit(df[f])
            # transform using LabelEncoder
            df.loc[:,f] = lbl_enc.transform(df[f])

    # get the training dataframe
    df_train = df[df.kfolds != fold].reset_index(drop=True)

    # get the validation dataframe
    df_val = df[df.kfolds == fold].reset_index(drop=True)

    # transform  training data
    x_train = df_train[features]

    # transform val data
    x_val = df_val[features]

    # initialize logistic regression module
    model = xgb.XGBClassifier()

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict prob on val set
    pred = model.predict_proba(x_val)[:,1]

    # compute AUC
    auc = metrics.roc_auc_score(df_val.income.values, pred)

    # print AUC
    print(auc)

if __name__ == "__main__":
    for i in range(5):
        run(i)
