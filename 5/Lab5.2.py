import pandas as pd
from datetime import time

df_ses = pd.read_csv("cinema_sessions.csv", delimiter=" ")
df_tit = pd.read_csv("titanic_with_labels.csv", delimiter=" ")

# 4
df_tit["age_kids"] = df_tit.loc[df_tit["age"] < 18, "age"]
df_tit["age_adults"] = df_tit.loc[((df_tit["age"] >= 18) & (df_tit["age"] < 50)), "age"]
df_tit["age_elder"] = df_tit.loc[df_tit["age"] > 50, "age"]
df_tit.loc[pd.notna(df_tit["age_kids"]), "age_kids"] = 1
df_tit.loc[pd.isna(df_tit["age_kids"]), "age_kids"] = "-"
df_tit.loc[pd.notna(df_tit["age_adults"]), "age_adults"] = 1
df_tit.loc[pd.isna(df_tit["age_adults"]), "age_adults"] = "-"
df_tit.loc[pd.notna(df_tit["age_elder"]), "age_elder"] = 1
df_tit.loc[pd.isna(df_tit["age_elder"]), "age_elder"] = "-"
df_tit = df_tit.drop("age", axis=1)
print(df_tit)

# 5
alc = []
drinks = df_tit["drink"].unique()
for drink in drinks:
    if ("beer" in drink) or ("пиво" in drink):
        alc.append(drink)

df_tit.loc[df_tit["drink"].isin(alc), "drink"] = 1
df_tit.loc[df_tit["drink"] != 1, "drink"] = 0
print(df_tit)

# 6
df_ses["session_start"] = pd.to_datetime(df_ses["session_start"], format="%H:%M:%S.000").dt.time
for item in ["morning", "day", "evening"]:
    df_tit[item] = 0
for check_number in df_tit["check_number"]:
    if time(4) <= df_ses.loc[check_number]["session_start"] < time(12):
        df_tit.loc[check_number, "morning"] = 1
    elif time(12) <= df_ses.loc[check_number]["session_start"] < time(17):
        df_tit.loc[check_number, "day"] = 1
    else:
        df_tit.loc[check_number, "evening"] = 1
print(df_tit)
