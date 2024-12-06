import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pylab as plt

df = pd.read_csv('/kaggle/input/tehran-house-prices-dataset/TehranHouse.csv')

df['Parking'] = df['Parking'].astype(int)
df['Warehouse'] = df['Warehouse'].astype(int)
df['Elevator'] = df['Elevator'].astype(int)
df['Area'] = df['Area'].astype(float)
df['Room'] = df['Room'].astype(int)
df['Price(USD)'] = df['Price(USD)'].astype(float)


X = df[['Area', 'Room', 'Parking','Warehouse','Elevator']]  
y = df['Price(USD)'] 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the KNeighborsRegressor model
knn_regressor = KNeighborsRegressor()

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],            # Test different numbers of neighbors
    'weights': ['uniform', 'distance'],     # Use uniform or distance-based weighting
    'p': [1, 2]                             # 1 = Manhattan distance, 2 = Euclidean distance
}

# Perform the grid search with cross-validation
grid_search = GridSearchCV(knn_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

# Fit the grid search model
grid_search.fit(X_train_scaled, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Print the accuracy (negative MSE) of each model
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(f"Parameters: {params}, Accuracy (Negative MSE): {mean_score}")

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"MAE: {mae}, RMSE: {rmse}, R^2: {r2}")

# Calculate mean and standard deviation for y_test and y_pred
y_test_mean = np.mean(y_test)
y_test_median = np.median(y_test)
y_test_std = np.std(y_test)
y_pred_mean = np.mean(y_pred)
y_pred_median = np.median(y_pred)
y_pred_std = np.std(y_pred)
'''
#Scatterplot actual vs predicted

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs. Actual Prices')
plt.show()
'''
# Create two subplots for y_test and y_pred histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the histogram for y_test
n_test, bins_test, patches_test = axes[0].hist(y_test, bins=20, color='blue', alpha=0.7)
axes[0].set_title('y_test (Actual Prices)')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')
axes[0].axvline(y_test_mean, color='r', linestyle='dashed', linewidth=2)
axes[0].text(y_test_mean + 10000, max(n_test) - 100 , f'Mean: {y_test_mean:.2f}\nSD: {y_test_std:.2f}', color='r')
axes[0].axvline(y_test_median, color='b', linestyle='solid', linewidth=2)
axes[0].text(y_test_median + 10000, max(n_test) - 200, f'Median: {y_test_median:.2f}', color='b')


# Plot the histogram for y_pred
n_pred, bins_pred, patches_pred = axes[1].hist(y_pred, bins=20, color='green', alpha=0.7)  # Only call once
axes[1].set_title('y_pred (Predicted Prices)')
axes[1].set_xlabel('Price')
axes[1].set_ylabel('Frequency')
axes[1].axvline(y_pred_mean, color='r', linestyle='dashed', linewidth=2)
axes[1].text(y_pred_mean + 10000, max(n_pred), f'Mean: {y_pred_mean:.2f}\nSD: {y_pred_std:.2f}', color='r')  # Use n_pred
axes[1].axvline(y_pred_median, color='g', linestyle='solid', linewidth=2)
axes[1].text(y_pred_median + 10000, max(n_pred) - 100, f'Median: {y_pred_median:.2f}', color='g')

# Set the same x-axis and y-axis limits for both subplots
x_min = min(min(bins_test), min(bins_pred))
x_max = max(max(bins_test), max(bins_pred))
y_max = max(max(n_test), max(n_pred))

# Apply the same limits to both axes
for ax in axes:
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, y_max])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

