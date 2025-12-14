
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import os

np.random.seed(42)

# Create synthetic dataset
N = 50_000

df = pd.DataFrame({
    "user_age": np.random.randint(18, 70, N),
    "user_activity_score": np.random.uniform(0, 1, N),
    "num_prev_clicks": np.random.poisson(2, N),
    "time_on_page": np.random.exponential(scale=30, size=N),
    "product_price": np.random.uniform(5, 500, N),
    "is_discounted": np.random.randint(0, 2, N),
    "category_ctr_mean": np.random.uniform(0.01, 0.15, N),
    "device_type": np.random.randint(0, 3, N), # 0=mobile,1=desktop,2=tablet
    "hour_of_day": np.random.randint(0, 24, N),
    "ad_quality_score": np.random.uniform(0.3, 0.9, N)
})

# CTR label generation (non-linear realistic formula)
linear = (
    0.01 * (70 - df.user_age) +
    2.5 * df.user_activity_score +
    0.1 * df.num_prev_clicks +
    0.003 * df.time_on_page +
    0.02 * df.is_discounted +
    5 * df.category_ctr_mean +
    0.5 * df.ad_quality_score -
    0.0005 * df.product_price
)

prob = 1 / (1 + np.exp(-linear))
df["clicked"] = np.random.binomial(1, prob)

X = df.drop("clicked", axis=1)
y = df["clicked"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model v1 (simpler)
model_v2 = xgb.XGBClassifier(
    n_estimators=80,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    use_label_encoder=False,
    eval_metric="logloss"
)
model_v2.fit(X_train, y_train)

# Train model v2 (improved)
model_v1 = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)
model_v1.fit(X_train, y_train)

# Evaluate
preds_v1 = model_v1.predict_proba(X_test)[:, 1]
preds_v2 = model_v2.predict_proba(X_test)[:, 1]

auc_v1 = roc_auc_score(y_test, preds_v1)
auc_v2 = roc_auc_score(y_test, preds_v2)

print(f"Model_v1 AUC = {auc_v1:.4f}")
print(f"Model_v2 AUC = {auc_v2:.4f}")

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(model_v1, "models/xgboost_model_v1.pkl")
joblib.dump(model_v2, "models/xgboost_model_v2.pkl")

print("Models saved in models/ directory.")
