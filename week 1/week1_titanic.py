"""
Week 1 - ML Fundamentals + Data Preprocessing
Machine Learning & AI Course

Assignments:
  1. Load dataset with Pandas, summarize basic stats (.info(), .describe())
  2. Handle missing data using mean/median imputation
  3. Encode categorical variables using LabelEncoder & OneHotEncoder

Mini Project 1: Titanic Survival Prediction - Data Cleaning Project
  - Clean missing data
  - Encode Sex, Embarked columns
  - Visualize age distribution (Matplotlib/Seaborn)
  - Output cleaned dataset as new CSV
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATASET_PATH = r"C:\Users\Home\Downloads\data\Titanic-Dataset.csv"
OUTPUT_DIR = r"C:\Users\Home\skillnexis\output"
CLEANED_CSV = os.path.join(OUTPUT_DIR, "Titanic-Dataset-Cleaned.csv")
AGE_PLOT = os.path.join(OUTPUT_DIR, "age_distribution.png")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("=" * 70)
    print("1. LOAD DATASET & BASIC STATS")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"\n--- .info() ---")
    df.info()
    print(f"\n--- .describe() ---")
    print(df.describe())
    print(f"\n--- Missing values per column ---")
    print(df.isnull().sum())
    print(f"\n--- Survival distribution ---")
    print(df["Survived"].value_counts(normalize=True).round(4))
    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("2. HANDLE MISSING DATA (imputation)")
    print("=" * 70)

    df = df.copy()

    # Age: median imputation (median is robust to outliers vs mean)
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    print(f"\nAge  : {df['Age'].isnull().sum()} missing -> median imputation (median={age_median})")

    # Embarked: only 2 missing -> mode imputation
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(f"Embarked : {df['Embarked'].isnull().sum()} missing -> mode imputation (mode={embarked_mode})")

    # Fare: 1 missing -> median imputation
    fare_median = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(fare_median)
    print(f"Fare : {df['Fare'].isnull().sum()} missing -> median imputation (median={fare_median})")

    # Cabin: ~77% missing -> drop column (not reliably imputable)
    print(f"Cabin : {df['Cabin'].isnull().sum()} missing ({df['Cabin'].isnull().mean():.1%}) -> column dropped")
    df = df.drop(columns=["Cabin"])

    print(f"\nRemaining missing values:\n{df.isnull().sum()}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("3. ENCODE CATEGORICAL VARIABLES")
    print("=" * 70)

    df = df.copy()

    # LabelEncoder for Sex (binary: male/female)
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    print(f"\nLabelEncoder -> Sex: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # OneHotEncoder for Embarked (nominal: S/C/Q)
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    embarked_encoded = ohe.fit_transform(df[["Embarked"]])
    embarked_df = pd.DataFrame(
        embarked_encoded,
        columns=[f"Embarked_{c}" for c in ohe.categories_[0]],
        index=df.index,
    )
    df = pd.concat([df, embarked_df], axis=1).drop(columns=["Embarked"])
    print(f"OneHotEncoder -> Embarked columns: {list(embarked_df.columns)}")

    # Drop columns not useful for modeling (keep PassengerId for reference)
    df = df.drop(columns=["Name", "Ticket"])

    print(f"\nFinal columns: {list(df.columns)}")
    return df


def visualize_age_distribution(original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("4. VISUALIZE AGE DISTRIBUTION")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(original["Age"].dropna(), bins=30, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Age Distribution BEFORE Imputation\n(missing={original['Age'].isnull().sum()})")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")

    sns.histplot(cleaned["Age"], bins=30, kde=True, ax=axes[1], color="darkorange")
    axes[1].set_title("Age Distribution AFTER Median Imputation")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(AGE_PLOT, dpi=150)
    plt.close()
    print(f"\nSaved plot -> {AGE_PLOT}")

    # Survival by sex (bonus quick view)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.countplot(data=original, x="Sex", hue="Survived", ax=ax2, palette="Set2")
    ax2.set_title("Survival Count by Sex")
    ax2.legend(title="Survived", labels=["No", "Yes"])
    survival_plot = os.path.join(OUTPUT_DIR, "survival_by_sex.png")
    plt.tight_layout()
    plt.savefig(survival_plot, dpi=150)
    plt.close()
    print(f"Saved plot -> {survival_plot}")


def train_test_split_demo(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("5. TRAIN / TEST SPLIT (80/20)")
    print("=" * 70)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Train survival rate: {y_train.mean():.3f} | Test survival rate: {y_test.mean():.3f}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset(DATASET_PATH)
    df = handle_missing_data(df)
    df = encode_categorical(df)
    visualize_age_distribution(pd.read_csv(DATASET_PATH), df)

    df.to_csv(CLEANED_CSV, index=False)
    print("\n" + "=" * 70)
    print("6. OUTPUT CLEANED DATASET")
    print("=" * 70)
    print(f"\nSaved cleaned CSV -> {CLEANED_CSV}")
    print(f"Cleaned shape: {df.shape}")
    print(f"\n--- Cleaned .describe() ---")
    print(df.describe())
    print(f"\n--- First 5 rows of cleaned data ---")
    print(df.head())

    train_test_split_demo(df)


if __name__ == "__main__":
    main()
