"""
Yuva Intern – Logistics Data Analyst – Week 4
Predictive Modeling and Optimization in Logistics
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load simulated data
df = pd.read_csv("week4_simulated_logistics_dataset.csv")

target = "Delivery_Time_Days"
features = [c for c in df.columns if c not in ["Shipment_ID", target]]
X = df[features]
y = df[target]

categorical_cols = ["Transport_Mode", "Traffic_Level", "Weather_Condition", "Priority"]
numeric_cols = [c for c in features if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=8, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=180, max_depth=12, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = []
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE_days": mae,
        "RMSE_days": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values("RMSE_days")
print(results_df.to_string(index=False))
print("\nBest model:", results_df.iloc[0]["Model"])

# Recommended next step for real operations:
# validate the selected model on time-based holdout data and monitor
# prediction error after deployment.
