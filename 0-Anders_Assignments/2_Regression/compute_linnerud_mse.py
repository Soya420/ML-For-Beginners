import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

rng = 42

data = load_linnerud()
X = data.data
y = data.target
X_situps = X[:, 1].reshape(-1, 1)
y_waist = y[:, 1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_situps, y_waist, test_size=0.25, random_state=rng)

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
cv_mse = -cross_val_score(LinearRegression(), X_situps, y_waist, cv=5, scoring='neg_mean_squared_error')

# Polynomial regression (degree=2)
degree = 2
poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)
mse_test_poly = mean_squared_error(y_test, y_pred_poly)
cv_mse_poly = -cross_val_score(make_pipeline(PolynomialFeatures(degree), LinearRegression()), X_situps, y_waist, cv=5, scoring='neg_mean_squared_error')

print(f"LinearRegression MSE (test): {mse_test:.6f}")
print(f"LinearRegression 5-fold CV MSE mean: {cv_mse.mean():.6f}, std: {cv_mse.std():.6f}")
print()
print(f"Polynomial degree={degree} MSE (test): {mse_test_poly:.6f}")
print(f"Polynomial degree={degree} 5-fold CV MSE mean: {cv_mse_poly.mean():.6f}, std: {cv_mse_poly.std():.6f}")
