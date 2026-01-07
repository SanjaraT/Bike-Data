import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = pd.read_csv("SeoulBikeData.csv", encoding="latin1").drop(["Date","Holiday","Seasons"],axis = 1)
#print(df.head())
#df.info()
df["Functioning Day"] = (df["Functioning Day"]=="Yes").astype(int)
#print(df.head())

#encoding categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]= le.fit_transform(df[col])

target = 'Rented Bike Count'

#scatter plot
# for col in df.columns:
#     if col != target:
#          plt.figure()
#          plt.scatter(df[col],df[target], alpha = 0.3)
#          plt.xlabel(col)
#          plt.ylabel(target)
#          plt.title(f"{col} vs {target}")
#          plt.show()

#correlation
corr_matrix = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix, annot=True, fmt = ".2f", cmap = "coolwarm")
plt.title("Correlation Matrix")
# plt.show()

df.drop(columns=["Dew point temperature","Wind speed","Visibility","Functioning Day"],inplace=True)
print(df.head())