# train.py

from sklearn import tree
from sklearn import metrics
import pandas as pd
import config, model_dispatcher
import joblib
import argparse

def run(df, foldno, model, dump_path):

    # training DataFrame
    train_df = df[df.kfold !=foldno].reset_index(drop=True)

    # validation DataFrame
    val_df = df[df.kfold ==foldno].reset_index(drop=True)

    x_train = train_df.drop("target", axis=1).values
    y_train = train_df.target.values

    x_valid = val_df.drop("target", axis=1).values
    y_valid = val_df.target.values

    # simple decision tree classifier
    clf = model_dispatcher.models[model]

    # fit the model
    clf = clf.fit(x_train, y_train)

    # compute accuracy on val data
    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Kfold:{foldno} accuracy on val set is: {accuracy}")

    # save the model
    joblib.dump(clf, dump_path + f"dt_{foldno}.bin")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--foldno", type=int, default=0,\
            help='enter between 0 and 4')
    parser.add_argument("--model", type=str,)

    arg = parser.parse_args()
    df = pd.read_csv(config.TRAINING_FILE)
    dump_path = config.MODEL_OUTPUT

    run(df, arg.foldno, arg.model, dump_path)
