# create_folds.py
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':

    # read train data
    df = pd.read_csv("../input/adult.data", header=None)
    df.columns =["age", "workclass", "fnlwgt", "education", "education.num", \
                 "marital.status", "occupation", "relationship", \
                 "race", "sex", "capital.gain", "capital.loss", \
                 "hours.per.week", "native.country", "income"]

    # create kfold column
    df["kfolds"] = -1

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.income.values

    # call stratified kfold class constructor
    kfold = model_selection.StratifiedKFold(n_splits=5)

    # fill in fold no in dataframe
    for foldno, (train_idx, val_idx) in enumerate(kfold.split(X=df, y=y)):
        df.loc[val_idx, "kfolds"] = foldno

    # save to csv
    df.to_csv("../input/train_folds.csv")
