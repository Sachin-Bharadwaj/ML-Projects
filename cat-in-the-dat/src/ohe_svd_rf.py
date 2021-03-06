# ohe_svd_rf.py

import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import decomposition
from scipy import sparse


def run(fold):

    df = pd.read_csv('../input/train_folds.csv')

    # find all feature names, all columns are features except id, target, kfolds
    features = [f for f in df.columns if f not in ["id", "target", "kfolds"]]

    # fill all NaNs with NONE
    # Note that converting to string, it does not matter because all are
    # categories in the feature set
    for f in features:
        df.loc[:,f] = df[f].astype(str).fillna("NONE")

    # get the training dataframe
    df_train = df[df.kfolds != fold].reset_index(drop=True)

    # get the validation dataframe
    df_val = df[df.kfolds == fold].reset_index(drop=True)

    # initialize one hot encoder
    ohe = preprocessing.OneHotEncoder()

    #fit ohe on training + val data
    full_data = pd.concat([df_train[features], df_val[features]], axis=0)
    ohe.fit(full_data)

    # transform  training data
    x_train = ohe.transform(df_train[features])

    # transform val data
    x_val = ohe.transform(df_val[features])

    # initialize truncated SVD
    svd = decomposition.TruncatedSVD(n_components=120)

    # fit svd on combined features
    full_sparse = sparse.vstack((x_train, x_val))
    svd.fit(full_sparse)

    # transform  training data
    x_train = svd.transform(x_train)

    # transform  training data
    x_val = svd.transform(x_val)

    # initialize logistic regression module
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # fit model on training data
    model.fit(x_train, df_train.target.values)

    # predict prob on val set
    pred = model.predict_proba(x_val)[:,1]

    # compute AUC
    auc = metrics.roc_auc_score(df_val.target.values, pred)

    # print AUC
    print(auc)

if __name__ == "__main__":
    for i in range(5):
        run(i)
