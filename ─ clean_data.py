import os
import pandas as pd

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/cleaned", exist_ok=True)

# ---------- LOAD ----------
raw = pd.read_csv(URL)
raw.to_csv("data/raw/titanic_raw.csv", index=False)
df = raw.copy()

# ---------- AUDIT (before) ----------
print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())
print("Duplicate PassengerIds:", df["PassengerId"].duplicated().sum())
print("\nData types:\n", df.dtypes)
print("\nSex values:", df["Sex"].unique())
print("Embarked values:", df["Embarked"].unique())

# ---------- CLEAN ----------
# 1. Standardise column names
df.columns = (df.columns.str.strip()
              .str.replace(r"(?<!^)(?=[A-Z])", "_", regex=True)
              .str.lower())  # PassengerId -> passenger_id, SibSp -> sib_sp

# 2. Trim whitespace and standardise text values
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()
df["sex"] = df["sex"].str.lower()
df["embarked"] = df["embarked"].str.upper().map(
    {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})

# 3. Remove duplicates
df = df.drop_duplicates().drop_duplicates(subset="passenger_id")

# 4. Fix data types
df["survived"] = df["survived"].astype("int8")
df["pclass"] = df["pclass"].astype("category")
for col in ["sex", "embarked"]:
    df[col] = df[col].astype("category")

# 5. Handle missing values
df["cabin_known"] = df["cabin"].notna()              # keep the signal
df = df.drop(columns="cabin")                        # ~77% missing, too sparse
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["age_imputed"] = df["age"].isna()                 # flag before filling
df["age"] = df["age"].fillna(
    df.groupby(["pclass", "sex"], observed=True)["age"].transform("median"))

# 6. Validate ranges
assert df["age"].between(0, 100).all()
assert (df["fare"] >= 0).all()
print("\nZero-fare rows (suspicious, kept):", (df["fare"] == 0).sum())

# ---------- AUDIT (after) ----------
print("\nFinal shape:", df.shape)
print("Remaining missing:", df.isna().sum().sum())
print(df.dtypes)

df.to_csv("data/cleaned/titanic_cleaned.csv", index=False)
