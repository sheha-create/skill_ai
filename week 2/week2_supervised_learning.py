"""
Week 2 - Supervised Learning (Regression & Classification)
Machine Learning & AI Course

Topics: Linear Regression, Logistic Regression, Decision Tree, Random Forest,
        Accuracy metrics (MSE, R2, Confusion Matrix)

Assignments:
  1. Build Linear Regression model on housing dataset (predict price)
  2. Train Logistic Regression on Titanic dataset for survival prediction

Mini Project 2: House Price Prediction Model
  - Train Linear Regression model
  - Predict house prices
  - Evaluate R2 score
  - Plot predicted vs actual values
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

HOUSING_PATH = r"C:\Users\Home\Downloads\data2\Housing.csv"
TITANIC_PATH = r"C:\Users\Home\skillnexis\week 1\output\Titanic-Dataset-Cleaned.csv"
OUTPUT_DIR = r"C:\Users\Home\skillnexis\week 2\output"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Assignment 1 + Mini Project 2: House Price Prediction (Linear Regression)
# ---------------------------------------------------------------------------

def load_housing() -> pd.DataFrame:
    section("1. LOAD HOUSING DATASET")
    df = pd.read_csv(HOUSING_PATH)
    print(f"Shape: {df.shape}")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())
    print("\n.describe():\n", df.describe())
    return df


def preprocess_housing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    section("2. PREPROCESS HOUSING DATA")
    df = df.copy()

    yes_no_cols = [
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning", "prefarea",
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})
    print(f"Label-mapped yes/no columns: {yes_no_cols}")

    le = LabelEncoder()
    df["furnishingstatus"] = le.fit_transform(df["furnishingstatus"])
    print(f"LabelEncoder -> furnishingstatus: "
          f"{dict(zip(le.classes_, le.transform(le.classes_)))}")

    print(f"Final columns: {list(df.columns)}")
    return df, yes_no_cols


def evaluate_regression(name: str, y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  MSE  = {mse:,.2f}")
    print(f"  RMSE = {rmse:,.2f}")
    print(f"  MAE  = {mae:,.2f}")
    print(f"  R2   = {r2:.4f}")
    return {"Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def housing_regression(df: pd.DataFrame) -> dict:
    section("3. TRAIN / TEST SPLIT (80/20)")
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    section("4. LINEAR REGRESSION (Mini Project 2)")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("\nCoefficients:")
    coef_df = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": lr.coef_}
    ).sort_values("Coefficient", ascending=False)
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: {lr.intercept_:,.2f}")

    results = [evaluate_regression("Linear Regression", y_test, y_pred_lr)]

    section("5. DECISION TREE & RANDOM FOREST (comparison)")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    results.append(evaluate_regression("Decision Tree", y_test, dt.predict(X_test)))

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_regression("Random Forest", y_test, rf.predict(X_test)))

    section("6. PLOT PREDICTED vs ACTUAL")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: predicted vs actual
    axes[0].scatter(y_test, y_pred_lr, alpha=0.6, edgecolors="k", linewidths=0.5)
    mn, mx = min(y_test.min(), y_pred_lr.min()), max(y_test.max(), y_pred_lr.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("Linear Regression: Predicted vs Actual House Price")
    axes[0].legend()

    # Residuals
    residuals = y_test - y_pred_lr
    axes[1].scatter(y_pred_lr, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
    axes[1].axhline(0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Price")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "predicted_vs_actual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")

    # Distribution of prices
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(y_test, bins=20, kde=True, color="steelblue", label="Actual", ax=ax2)
    sns.histplot(y_pred_lr, bins=20, kde=True, color="darkorange", alpha=0.5,
                 label="Predicted", ax=ax2)
    ax2.set_title("Distribution: Actual vs Predicted Prices")
    ax2.legend()
    path2 = os.path.join(OUTPUT_DIR, "price_distribution.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved -> {path2}")

    return {"y_test": y_test, "y_pred": y_pred_lr, "results": results}


# ---------------------------------------------------------------------------
# Assignment 2: Logistic Regression on Titanic
# ---------------------------------------------------------------------------

def titanic_logistic() -> None:
    section("7. LOGISTIC REGRESSION ON TITANIC (survival prediction)")
    df = pd.read_csv(TITANIC_PATH)
    print(f"Shape: {df.shape}")

    X = df.drop(columns=["Survived", "PassengerId"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Died", "Survived"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Titanic Logistic Regression - Confusion Matrix")
    path = os.path.join(OUTPUT_DIR, "titanic_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_housing()
    df, _ = preprocess_housing(df)
    reg_out = housing_regression(df)

    section("8. MODEL COMPARISON SUMMARY")
    print(pd.DataFrame(reg_out["results"]).to_string(index=False))

    titanic_logistic()


if __name__ == "__main__":
    main()
