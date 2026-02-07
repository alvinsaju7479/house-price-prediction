
import joblib
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/raw/housing.csv"
MODEL_PATH = "models/model.pkl"
OUT_PATH = "reports/figures/feature_importance.png"
TARGET = "price"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])

pipe = joblib.load(MODEL_PATH)

pre = pipe.named_steps["preprocessor"]
model = pipe.named_steps["model"]

# Get feature names after preprocessing (numeric + onehot categorical)
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

ohe = pre.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + cat_feature_names

importances = model.feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
fi.sort_values().plot(kind="barh")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
print(f"✅ Saved: {OUT_PATH}")
