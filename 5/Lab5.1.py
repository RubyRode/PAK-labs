import pandas as pd
df_titanic = pd.read_csv("titanic_with_labels.csv", delimiter=" ")

# 1
empty_titanic = df_titanic[((df_titanic["sex"] == "Не указан") | (df_titanic["sex"] == "-"))].index
df_titanic = df_titanic.drop(index=empty_titanic)
df_titanic.loc[((df_titanic["sex"] == "ж") | (df_titanic["sex"] == "Ж")), 'sex'] = 0
df_titanic.loc[((df_titanic["sex"] == "м") | (df_titanic["sex"] == "М") | (df_titanic["sex"] == "M")) | (df_titanic["sex"] == "Мужчина"), 'sex'] = 1

# 2
df_titanic.loc[pd.isna(df_titanic["row_number"]), "row_number"] = max(df_titanic['row_number'])

# 3
mean_liters = df_titanic[((df_titanic["liters_drunk"] <= 5) & (df_titanic["liters_drunk"] >= 0))].liters_drunk.mean()
df_titanic.loc[((df_titanic["liters_drunk"] > 5) | (df_titanic["liters_drunk"] < 0)), "liters_drunk"] = int(mean_liters)

