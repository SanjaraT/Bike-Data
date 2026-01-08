import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# print(df.head())

#train-test-validation
X = df.drop(columns = ['Rented Bike Count'])
y = df['Rented Bike Count']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

#standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns= X_train.columns)
print(X_train_scaled_df)