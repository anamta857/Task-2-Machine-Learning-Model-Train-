import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("dataset.csv")

print("Dataset Shape:")
print(df.shape)

# Missing Values Check
print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check Duplicates
print("\nDuplicates:", df.duplicated().sum())

# Remove Duplicates
df.drop_duplicates(inplace=True)

print("\nFinal Shape:")
print(df.shape)

# Save Cleaned Dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved successfully!")

# Feature Selection
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)

print("\n===== Linear Regression =====")
print("R2 Score:", lr_r2)
print("MSE:", lr_mse)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_r2 = r2_score(y_test, y_pred_dt)
dt_mse = mean_squared_error(y_test, y_pred_dt)

print("\n===== Decision Tree =====")
print("R2 Score:", dt_r2)
print("MSE:", dt_mse)

# Best Model
print("\n===== Model Comparison =====")

if lr_r2 > dt_r2:
    print("Best Model: Linear Regression")
else:
    print("Best Model: Decision Tree Regressor")