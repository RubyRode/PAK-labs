from datetime import date

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
df = pd.read_csv("wells_info.csv")


# 1
dataframe = np.random.rand(10, 5)
dataframe = pd.DataFrame(dataframe)
print(dataframe)
res = dataframe[dataframe >= 0.3].mean(axis=1)
print(res)

# 2
start_column = "SpudDate"
df[start_column] = pd.to_datetime(df[start_column], format="%Y-%m-%d")
df["CompletionDate"] = pd.to_datetime(df["CompletionDate"], format="%Y-%m-%d")
df["Delta"] = ((df["CompletionDate"] - df[start_column]) / np.timedelta64(1, 'M')).astype(int)
# print(df)

# 3
d = pd.read_csv("wells_info_na.csv")
df2 = d.copy()
df2 = df2.fillna(0)

for column, dtype in d.dtypes.items():
    if dtype == np.dtype['int64']:
        df2.loc[df2[column] == 0, column] = d.loc[d[column] != np.NaN, column].mean()
    else:
        df2.loc[df2[column] == 0, column] = d[column].value_counts().sort_values(ascending=False).index[0]

# print(df2)

# 4
df1 = pd.read_csv("wells_info.csv")
df2 = pd.read_csv("production.csv")
for API in df1["API"]:
    df2_sort = df2.loc[df2["API"] == API].sort_values(["Year", "Month"])
    index = df1.loc[df1["API"] == API].index
    df1.loc[index, "Sum12"] = 0
    df1.loc[index, "Sum"] = 0
    for item in ("Liquid", "Gas", "Water"):
        df1.loc[index, f"Sum12{item}"] = df2_sort[0:12][item].sum()
        df1.loc[index, f"Sum{item}"] = df2_sort[item].sum()
        df1.loc[index, f"Sum12"] += df1.loc[index, f"Sum12{item}"]
        df1.loc[index, f"Sum"] += df1.loc[index, f"Sum{item}"]
# print(df1)
