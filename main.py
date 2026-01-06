import pandas as pd

df = pd.read_csv("SeoulBikeData.csv", encoding="latin1").drop(["Date","Holiday","Seasons"],axis = 1)
#print(df.head())
df.info()
df["Functioning Day"] = (df["Functioning Day"]=="Yes").astype(int)
print(df.head())