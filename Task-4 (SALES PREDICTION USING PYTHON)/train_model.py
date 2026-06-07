import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load Dataset
df = pd.read_csv("advertising.csv")

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("RMSE:", round(mean_squared_error(y_test, y_pred)**0.5, 4))

# Save Model
joblib.dump(model, "sales_model.pkl")

print("Model saved as sales_model.pkl")