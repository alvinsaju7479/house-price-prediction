
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"# 🏠 House Price Prediction — EDA\n"
"This notebook explores the housing dataset and highlights patterns that influence `price`."
))

cells.append(nbf.v4.new_code_cell(
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n\n"
"df = pd.read_csv('../data/raw/housing.csv')\n"
"df.head()"
))

cells.append(nbf.v4.new_code_cell("df.info()"))
cells.append(nbf.v4.new_code_cell("df.isna().sum().sort_values(ascending=False).head(20)"))

cells.append(nbf.v4.new_code_cell(
"plt.figure(figsize=(6,4))\n"
"df['price'].hist(bins=30)\n"
"plt.title('Target Distribution: price')\n"
"plt.xlabel('price')\n"
"plt.ylabel('count')\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(nbf.v4.new_code_cell(
"numeric_cols = df.select_dtypes(include=['int64','float64']).columns\n"
"corr = df[numeric_cols].corr()['price'].sort_values(ascending=False)\n"
"corr"
))

cells.append(nbf.v4.new_code_cell(
"plt.figure(figsize=(6,4))\n"
"plt.scatter(df['area'], df['price'], alpha=0.5)\n"
"plt.title('area vs price')\n"
"plt.xlabel('area')\n"
"plt.ylabel('price')\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(nbf.v4.new_code_cell(
"plt.figure(figsize=(6,4))\n"
"df.groupby('furnishingstatus')['price'].mean().sort_values().plot(kind='bar')\n"
"plt.title('Avg price by furnishingstatus')\n"
"plt.xlabel('furnishingstatus')\n"
"plt.ylabel('avg price')\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
"## Key EDA Notes (fill these)\n"
"- Which features correlate most with price?\n"
"- Any strong categorical splits (airconditioning, furnishingstatus, prefarea)?\n"
"- Any outliers in area/price?\n"
))

nb["cells"] = cells

out = Path("notebooks/01_eda.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(nbf.writes(nb), encoding="utf-8")

print("✅ Recreated notebooks/01_eda.ipynb")

