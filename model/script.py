import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


## Data Cleaning 

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

data = pd.read_excel("opticalcData.xlsx")
data.drop(columns=["ID", "LT (mm)", "WTW", "TK1", "TK2", "CCT "], axis=1, inplace=True)
data.loc[data["Gender"] == 'F', 'Gender'] = 0
data.loc[data["Gender"] == 'M', 'Gender'] = 1
data.loc[data["Eye"] == 'OS', 'Eye'] = 0
data.loc[data["Eye"] == 'OD', 'Eye'] = 1
data.loc[(data["K or TK"] == 'K'), 'K or TK'] = 0
data.loc[(data["K or TK"] == 'K  '), 'K or TK'] = 0
data.loc[data["K or TK"] == 'TK', 'K or TK'] = 1
data.loc[(data["K or TK"] == "TK  "), 'K or TK'] = 1


data = data.apply(pd.to_numeric, errors='coerce')
# drop rows with missing data
data.dropna(axis=0, how="any", inplace=True)


## PCA Set Up
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)
# pca = PCA()
# pca.fit(scaled_data)
# pca_data = pca.transform(scaled_data)
# # print("PCA Components:\n", pca.components_)
# # print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# # print("PCA Transformed Data:\n", pca_data.shape)


# components = pca.components_
# feature_names = data.columns
# # Print the components
# for i, component in enumerate(components):
#     print(f"Component {i+1}:")
#     for feature, value in zip(feature_names, component):
#         print(f"{feature}: {value:.5f}")
#     print()

# print(data.columns)
# correlation_matrix = data.corr(method='pearson')
# print(correlation_matrix["PREDICTION ERROR (outcome SE - Predicted SE"].sort_values(ascending=False))

X = data[['Predicted SE/refraction\n(calculator output prediction)', 'Age', 'Q', 'Gender', 'SA', 'K1', 'K2']]
y = data['Outcome SE/refraction\n(after surgery vision)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)

# Predict on the test set
y_pred = random_forest_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# importances = random_forest_model.feature_importances_
# print("Feature Importances:")
# for feature, importance in zip(X.columns, importances):
#     print(f"{feature}: {importance}")

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse}")