# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset and perform initial inspection; remove irrelevant columns and convert categorical variables into numerical form using one-hot encoding.
2. Separate the dataset into independent features (X) and target variable (y), and apply standard scaling to normalize the data.
3. Split the scaled data into training and testing sets using an 80:20 ratio.
4. Initialize the Stochastic Gradient Descent (SGD) Regressor and train the model using the training data.
5. Predict car prices on the test data, evaluate the model using MSE, MAE, and R² score, and visualize actual versus predicted values.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: SHANTHOSH KUMAR R
RegisterNumber:  212225040402
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Name: SHANTHOSH KUMAR R")
print("Reg. No:212225040402")
print("MSE (Mean Squared Error):", mse)
print("R-squared Score:", r2)
print("MAE (Mean Absolute Error):",mae)
print("Model Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)],color='red')
plt.show()

```

## Output:
<img width="951" height="625" alt="image" src="https://github.com/user-attachments/assets/c54306e2-817c-4d34-99f2-9dfbc5ee7c79" />
<img width="623" height="723" alt="image" src="https://github.com/user-attachments/assets/fabc986a-0408-4385-881a-5b0662e82e26" />
<img width="1245" height="406" alt="image" src="https://github.com/user-attachments/assets/f062ae19-160c-4a13-84d9-f5c797ce1b90" />
<img width="835" height="556" alt="image" src="https://github.com/user-attachments/assets/26d9d1ca-0945-420d-8416-0e901a0cae09" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
