import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load data and split into training (65%) and testing (35%)
data = pd.read_csv("asianpaint.csv", parse_dates=["Date"], index_col="Date")

# Split data
train_size = int(len(data) * 0.65)
train_data = data.iloc[:train_size, 0]  # First 65% of data
test_data = data.iloc[train_size:, 0]   # Remaining 35% of data

# Plot train and test data
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Train Data')
plt.plot(test_data, label='Test Data')
plt.title('Asian Paints Stock Prices (Train and Test Data)')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.legend()
plt.show()

# Step 2: Build an Autoregression Model with one-day lag
def autoregression(train_data):
    # Prepare lagged data: x(t) predicts x(t+1)
    X = train_data[:-1]  # X(t)
    y = train_data[1:]   # X(t+1)
    
    # Add bias (intercept) term
    X = np.vstack([np.ones(len(X)), X]).T
    
    # Solve for weights using normal equation: w = (X^T * X)^-1 * X^T * y
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

# Train the AR model and find the coefficients
w0, w1 = autoregression(train_data)
print(f"Autoregression coefficients: w0 = {w0}, w1 = {w1}")

# Step 3: Make predictions for the test data
def predict(test_data, w0, w1):
    predictions = []
    for i in range(len(test_data) - 1):
        # Use the previous day's actual price to predict the next day's price
        pred = w0 + w1 * test_data.iloc[i]
        predictions.append(pred)
    return np.array(predictions)

# Get predictions
predictions = predict(test_data, w0, w1)
actual_test = test_data.iloc[1:].values  # Test data, excluding the first value

# Step 4: Evaluate the model using RMSE and MAPE
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

rmse_value = rmse(actual_test, predictions)
mape_value = mape(actual_test, predictions)

print(f"RMSE (%): {rmse_value}")
print(f"MAPE (%): {mape_value}")

# Step 5: Plot the actual vs predicted values for the test data
plt.figure(figsize=(10, 6))
plt.plot(test_data.index[1:], actual_test, label='Actual Test Data', color='orange')
plt.plot(test_data.index[1:], predictions, label='Predicted Test Data', color='blue')
plt.title('Actual vs Predicted Test Data')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.legend()
plt.show()
