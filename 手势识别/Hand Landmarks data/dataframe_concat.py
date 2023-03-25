import pandas as pd
import os


def dataframe_concat(n):
    dataframe = None
    for i in range(1, n + 1):
        filePath = f"Hand Landmarks data/label {i}.csv"
        dataframe = pd.concat([dataframe, pd.read_csv(filePath, index_col=0)], ignore_index=True)
    dataframe.to_csv("Hand Landmarks data/final.csv")


if __name__ == "__main__":
    n = len(os.listdir('./')) - 1
    dataframe_concat(n)
