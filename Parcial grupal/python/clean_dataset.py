import os
import pandas as pd

data = pd.read_csv("./chronic_kidney_disease.csv", on_bad_lines='skip')

data = data.replace('?', pd.NA)

data = data.drop(columns=['rbc'])

categorical_columns = ["sg", "al", "su", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
for col in categorical_columns:
    most_frequent = data[col].mode()[0] if not data[col].mode().empty else None
    data[col] = data[col].fillna(most_frequent)

numeric_columns = ["age", "bgr", "bp", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    if col in ["age", "rbcc"]:
        data[col] = data[col].fillna(data[col].mean())
    else:
        data[col] = data[col].fillna(data[col].median())

integer_columns = ["age", "bp", "bgr", "bu", "sc", "pcv", "wbcc"]
for col in integer_columns:
    data[col] = data[col].astype(int)

data.to_csv("processed_ckd_data.csv", index=False)
