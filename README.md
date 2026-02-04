# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.

2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.

3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.

4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SRI SAI SARAN G
RegisterNumber:  212225220103
*/

from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

# Sample data (2 features)
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])

# Create model
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')

# Train model
model.fit(X, y)

# Check learned weights
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Predict
y_pred = model.predict(X)

# Plot Actual vs Predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.show()

```

## Output:
<img width="551" height="39" alt="JB" src="https://github.com/user-attachments/assets/bb0a1126-dd5a-4a84-9ee5-688340f265d2" />

<img width="586" height="377" alt="543602731-7109a08a-9ef4-4efd-923f-85de7e622578" src="https://github.com/user-attachments/assets/553ba738-1faf-47b9-8be2-70ff2c5711e7" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
