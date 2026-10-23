"""
Week 4 - CAPSTONE: Loan Approval System
Machine Learning & AI Course

Workflows:
  1. Clean categorical data (Gender, Marital Status, Income)
  2. Train Random Forest / XGBoost model
  3. Evaluate with Confusion Matrix
  4. Predict if loan can be approved

Outcome: Practical classification problem.
"""

import os
import pickle

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_PATH = r"C:\Users\Home\Downloads\week 4 data\train_u6lujuX_CVtuZ9i (1).csv"
OUTPUT_DIR = r"C:\Users\Home\skillnexis\week 4\output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "loan_data_cleaned.csv")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    section("1. LOAD DATASET & BASIC STATS")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    df.info()
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nLoan approval rate: {df['Loan_Status'].value_counts(normalize=True).round(4).to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 2. Clean categorical data (Gender, Marital Status, Income)
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    section("2. CLEAN CATEGORICAL DATA (Gender, Marital Status, Income)")
    df = df.copy()

    # --- Categorical imputation (mode) ---
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
    for col in cat_cols:
        mode = df[col].mode()[0]
        n = df[col].isnull().sum()
        df[col] = df[col].fillna(mode)
        print(f"  {col:<16}: {n} missing -> mode imputation ({mode})")

    # --- Numerical / income-related imputation ---
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    print(f"  {'LoanAmount':<16}: median imputation ({df['LoanAmount'].median()})")

    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
    print(f"  {'Loan_Amount_Term':<16}: median imputation ({df['Loan_Amount_Term'].median()})")

    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    print(f"  {'Credit_History':<16}: mode imputation ({df['Credit_History'].mode()[0]})")

    # --- Income cleaning: combined total income, log-scale to reduce skew ---
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    print(f"\n  Income feature: TotalIncome = ApplicantIncome + CoapplicantIncome")
    print(f"  TotalIncome range: {df['TotalIncome'].min():,.0f} - {df['TotalIncome'].max():,.0f} "
          f"(median {df['TotalIncome'].median():,.0f})")
    print(f"  Log-transform applied (log1p) to stabilize skew.\n")
    print(f"Remaining missing values:\n{df.isnull().sum()}")
    return df


# ---------------------------------------------------------------------------
# 3. Encode categoricals
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    section("3. ENCODE CATEGORICAL FEATURES")
    df = df.copy()
    encoders = {}

    # Ordinal: Dependents 0/1/2/3+
    df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

    # Label-encode binary categoricals
    binary_cols = ["Gender", "Married", "Education", "Self_Employed", "Loan_Status"]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  LabelEncoder {col:<16}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # One-hot: Property_Area
    dummies = pd.get_dummies(df["Property_Area"], prefix="Property_Area", dtype=int)
    df = pd.concat([df, dummies], axis=1).drop(columns=["Property_Area"])
    print(f"  OneHotEncoder Property_Area : {list(dummies.columns)}")

    print(f"\nFinal feature columns: {list(df.columns)}")
    return df, encoders


# ---------------------------------------------------------------------------
# 4. Train Random Forest / XGBoost
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame):
    section("4. TRAIN / TEST SPLIT + MODELS (Random Forest / XGBoost)")
    drop_cols = ["Loan_ID", "Loan_Status"]
    X = df.drop(columns=drop_cols)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Features ({X.shape[1]}): {list(X.columns)}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, scale_pos_weight=1.0,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"\n{name} trained.")

    return X_train, X_test, y_test, trained, X.columns


# ---------------------------------------------------------------------------
# 5. Evaluate with Confusion Matrix
# ---------------------------------------------------------------------------

def evaluate(X_test, y_test, trained, feature_names) -> pd.DataFrame:
    section("5. EVALUATION - CONFUSION MATRIX & METRICS")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, model) in zip(axes, trained.items()):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1": f1, "ROC-AUC": auc})

        print(f"\n{name}:")
        print(f"  Accuracy  = {acc:.4f}")
        print(f"  Precision = {prec:.4f}")
        print(f"  Recall    = {rec:.4f}")
        print(f"  F1        = {f1:.4f}")
        print(f"  ROC-AUC   = {auc:.4f}")
        print(f"  Classification report:\n{classification_report(y_test, y_pred, target_names=['Rejected (N)', 'Approved (Y)'])}")
        print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=["Rejected", "Approved"]) \
            .plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} (acc={acc*100:.1f}%)")

    plt.suptitle("Loan Approval - Confusion Matrix", y=1.02, fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")

    # ROC curves
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for name, model in trained.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax2.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves - Loan Approval Models")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    path2 = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved -> {path2}")

    # Feature importance (both models)
    fig3, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (name, model) in zip(axes, trained.items()):
        imp = pd.Series(model.feature_importances_, index=feature_names) \
            .sort_values(ascending=True)
        imp.plot(kind="barh", ax=ax, color="teal" if name == "Random Forest" else "darkorange")
        ax.set_title(f"{name} - Feature Importance")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Saved -> {path3}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Predict if loan can be approved
# ---------------------------------------------------------------------------

def predict_demo(model, encoders, feature_names) -> None:
    section("6. PREDICT: CAN THE LOAN BE APPROVED?")

    applicants = [
        # (Gender, Married, Dependents, Education, Self_Employed,
        #  ApplicantIncome, CoapplicantIncome, LoanAmount, Term, Credit_History, Property_Area)
        ("Male",   "Yes", 0, "Graduate",     "No",  5000, 2000, 120, 360, 1.0, "Semiurban"),
        ("Female", "No",  0, "Not Graduate", "Yes", 2500,    0,  80, 360, 0.0, "Rural"),
        ("Male",   "Yes", 2, "Graduate",     "No",  8000, 1500, 200, 180, 1.0, "Urban"),
        ("Female", "Yes", 1, "Graduate",     "No",  4000, 1000, 100, 360, 1.0, "Semiurban"),
        ("Male",   "No",  0, "Not Graduate", "No",  1500,    0,  50, 360, 0.0, "Urban"),
    ]

    rows = []
    for a in applicants:
        (gender, married, deps, edu, self_emp, inc, co_inc,
         loan_amt, term, credit, prop_area) = a

        total_income = inc + co_inc
        row = {
            "Gender": encoders["Gender"].transform([gender])[0],
            "Married": encoders["Married"].transform([married])[0],
            "Dependents": int(deps),
            "Education": encoders["Education"].transform([edu])[0],
            "Self_Employed": encoders["Self_Employed"].transform([self_emp])[0],
            "ApplicantIncome": inc,
            "CoapplicantIncome": co_inc,
            "LoanAmount": loan_amt,
            "Loan_Amount_Term": term,
            "Credit_History": credit,
            "TotalIncome": total_income,
            "LogTotalIncome": np.log1p(total_income),
            "LogLoanAmount": np.log1p(loan_amt),
        }
        for area in ["Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"]:
            row[area] = int(prop_area in area)

        X_new = pd.DataFrame([row])[list(feature_names)]
        proba = model.predict_proba(X_new)[0][1]
        pred = model.predict(X_new)[0]
        rows.append({
            "Gender": gender, "Married": married, "Education": edu,
            "Income": total_income, "LoanAmount": loan_amt,
            "Credit_History": credit, "Property": prop_area,
            "Approval Prob": f"{proba:.2%}",
            "Decision": "APPROVED" if pred == 1 else "REJECTED",
        })

    result = pd.DataFrame(rows)
    print("\n" + result.to_string(index=False))
    print("\nDecision rule: model class 1 = Approved (Y), class 0 = Rejected (N)")
    path = os.path.join(OUTPUT_DIR, "predictions_demo.csv")
    result.to_csv(path, index=False)
    print(f"Saved -> {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    df, encoders = encode_features(df)

    df.to_csv(CLEANED_CSV, index=False)
    print(f"\nCleaned dataset saved -> {CLEANED_CSV}")

    X_train, X_test, y_test, trained, feature_names = train_models(df)
    metrics = evaluate(X_test, y_test, trained, feature_names)

    section("7. MODEL COMPARISON")
    print(metrics.round(4).to_string(index=False))
    best_name = metrics.loc[metrics["ROC-AUC"].idxmax(), "Model"]
    print(f"\nBest model: {best_name} (highest ROC-AUC)")

    # Save best model + preprocessing artifacts
    artifacts = {"model": trained[best_name], "encoders": encoders,
                 "feature_names": list(feature_names), "best_model_name": best_name}
    joblib_path = os.path.join(MODEL_DIR, "loan_approval_model.joblib")
    joblib.dump(artifacts, joblib_path)
    pkl_path = os.path.join(MODEL_DIR, "loan_approval_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nSaved model -> {joblib_path}")
    print(f"Saved model -> {pkl_path}")

    predict_demo(trained[best_name], encoders, feature_names)


if __name__ == "__main__":
    main()
