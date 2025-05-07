import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()

    return df


# def create(df, features):
    # print(df.columns)
    # print(df.iloc[0]['malignancy'])

    # df['label'] = (df['malignancy'] >= 3).astype(int)
    # df['label'].astype(int)

    # return df


# def preprocess_train(df, features):
#     # print(df.columns)
#     # print(df.iloc[0]['malignancy'])

#     df['label'] = (df['malignancy'] >= 3).astype(int)

#     # features = df.columns
#     feat = df[features]
#     target = df['label']

#     X_train



# def load_data(filep):
#     filep