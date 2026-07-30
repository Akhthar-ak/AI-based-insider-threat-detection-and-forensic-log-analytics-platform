import pandas as pd

logon = pd.read_csv("data/raw/logon.csv")

print(logon.head())
print(logon.columns)
print(logon.info())