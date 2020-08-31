# create_folds.py

from sklearn import datasets
from sklearn import model_selection
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # download the dataset from sklearn
    data = datasets.fetch_openml(name='mnist_784', version=1, return_X_y=True)
    # this is tuple
    features, target = data
    # convert target from string to numpy integer
    target = target.astype(np.int)

    # create dataframe
    data_df = pd.DataFrame(np.column_stack((features,target)),\
     columns=[f"f_{i}" for i in range(features.shape[1])] + ['target'])

    # add a column kfold to DataFrame
    data_df["kfold"] = -1

    # randomly shuffle the data
    # shuffling destroys the index order so drop the index
    # reset_index() removes the index and adds as column to DataFrame
    # so give drop=True to remove the added index
    data_df.sample(frac=1).reset_index(drop=True)


    # initiate the kfold class
    kf = model_selection.KFold(n_splits=5)

    for foldno, (trn_idx, val_idx) in enumerate(kf.split(X=data_df)):
        data_df.loc[val_idx,"kfold"] = foldno

    # save the DataFrame
    data_df.to_csv("../input/train_folds.csv", index=False)
