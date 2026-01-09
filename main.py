import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# print(X_train_scaled_df)

#Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

#evaluation
# print("Train Set")
# print("RMSE :", np.sqrt(mean_squared_error(y_train, y_train_pred)))
# print("MAE :", mean_absolute_error(y_train, y_train_pred))
# print("R2 :", r2_score(y_train, y_train_pred))

# print("\nValidation Set")
# print("RMSE :", np.sqrt(mean_squared_error(y_val,y_val_pred)))
# print("MAE :", mean_absolute_error(y_val,y_val_pred))
# print("R2 :", r2_score(y_val,y_val_pred))

# print("\nTest Set")
# print("RMSE :", np.sqrt(mean_squared_error(y_test,y_test_pred)))
# print("MAE :", mean_absolute_error(y_test,y_test_pred))
# print("R2 :", r2_score(y_test,y_test_pred))

# plt.figure()
# plt.scatter(y_test, y_test_pred)
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],color = 'red')
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted (Best Fit Line)")
# plt.show()

#NEURAL NET LINEAR REGRESSION
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

# nn_model = Sequential([Dense(1,input_shape =(X_train_scaled.shape[1],))])

# nn_model.compile(
#     optimizer = Adam(learning_rate = 0.01),
#     loss = 'mse',
#     metrics = [MeanAbsoluteError()]
# )

# history = nn_model.fit(
#     X_train_scaled, y_train,
#     validation_data = (X_val_scaled, y_val),
#     epochs = 1000,
#     batch_size = 32,
#     verbose = 1
# )

# train_loss, train_mae = nn_model.evaluate(X_train_scaled, y_train, verbose=0)
# val_loss, val_mae     = nn_model.evaluate(X_val_scaled, y_val, verbose=0)
# test_loss, test_mae   = nn_model.evaluate(X_test_scaled, y_test, verbose=0)

# print("Neural Net Linear Regression")
# print(f"Train RMSE: {train_loss**0.5}")
# print(f"Val RMSE  : {val_loss**0.5}")
# print(f"Test RMSE : {test_loss**0.5}")

#DEEP NEURAL NETWORK
dnn_model = Sequential([
    Dense(128, activation = 'relu', input_shape = (X_train_scaled.shape[1],)),
    Dense(64, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(1)
])

dnn_model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = 'mse',
    metrics = ['mae']
)
early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 15,
    restore_best_weights = True
)
