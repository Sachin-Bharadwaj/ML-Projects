# test_folds.py

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../input/train_folds.csv')

    print(df.kfolds.value_counts())

    # target ditribution per foldno
    for k in range(0,5):
        print(f"fold no:{k} distribution: {df[df.kfolds == k].target.value_counts()}")
