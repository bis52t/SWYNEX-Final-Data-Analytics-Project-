import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
df = pd.read_csv("data/titanic_cleaned.csv")

# ---------- 1. KEY STATISTICS ----------
print("Shape:", df.shape)
print("\nNumeric summary:\n", df[["age", "fare", "sib_sp", "parch"]].describe().round(2))
print("\nOverall survival rate: {:.1%}".format(df["survived"].mean()))
df["family_size"] = df["sib_sp"] + df["parch"] + 1
df["family_group"] = pd.cut(df["family_size"], [0, 1, 4, 20],
                            labels=["Alone", "Small (2-4)", "Large (5+)"])
real_age = df[~df["age_imputed"]].copy()      # exclude imputed ages from age analysis
real_age["age_group"] = pd.cut(real_age["age"], [0, 12, 18, 35, 60, 100],
                               labels=["Child", "Teen", "Young adult", "Adult", "Senior"])

def rate(col, data=df):
    t = data.groupby(col, observed=True)["survived"].agg(["mean", "count"])
    t["mean"] = (t["mean"] * 100).round(1)
    t.columns = ["survival_%", "passengers"]
    print(f"\nSurvival by {col}:\n", t)
    return t

by_sex, by_class = rate("sex"), rate("pclass")
by_family, by_port = rate("family_group"), rate("embarked")
by_age = rate("age_group", real_age)
by_cabin = rate("cabin_known")

# ---------- 2. ANOMALIES ----------
q1, q3 = df["fare"].quantile([.25, .75])
upper = q3 + 1.5 * (q3 - q1)
outliers = df[df["fare"] > upper]
print(f"\nFare outliers (> {upper:.2f}): {len(outliers)}; max fare = {df['fare'].max():.2f}")
print("Zero-fare passengers:", (df["fare"] == 0).sum())
print("Rows with imputed age: {:.1%}".format(df["age_imputed"].mean()))

# ---------- 3. CHARTS ----------
def bar(t, title, fname):
    ax = t["survival_%"].plot(kind="bar", color="#2a6f97", figsize=(6, 4), rot=0)
    ax.set_title(title); ax.set_ylabel("Survival rate (%)"); ax.set_ylim(0, 100)
    for i, v in enumerate(t["survival_%"]):
        ax.text(i, v + 1, f"{v}%", ha="center")
    plt.tight_layout(); plt.savefig(f"charts/{fname}.png", dpi=150); plt.close()

bar(by_sex, "Survival rate by sex", "1_survival_by_sex")
bar(by_class, "Survival rate by passenger class", "2_survival_by_class")
bar(by_age, "Survival rate by age group", "3_survival_by_age")
bar(by_family, "Survival rate by family size", "4_survival_by_family")
bar(by_port, "Survival rate by port of embarkation", "5_survival_by_port")

plt.figure(figsize=(6, 4))
plt.hist([real_age[real_age.survived == 0]["age"], real_age[real_age.survived == 1]["age"]],
         bins=15, stacked=True, label=["Died", "Survived"], color=["#c1121f", "#2a9d8f"])
plt.title("Age distribution (known ages only)"); plt.xlabel("Age"); plt.legend()
plt.tight_layout(); plt.savefig("charts/6_age_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
df.boxplot(column="fare", by="pclass", grid=False)
plt.title("Fare by class (outliers shown)"); plt.suptitle(""); plt.xlabel("Class"); plt.ylabel("Fare")
plt.tight_layout(); plt.savefig("charts/7_fare_by_class.png", dpi=150); plt.close()

corr = df[["survived", "pclass", "age", "fare", "family_size"]].corr().round(2)
print("\nCorrelations:\n", corr)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, corr.iloc[i, j], ha="center", va="center", fontsize=8)
plt.colorbar(im); plt.title("Correlation matrix"); plt.tight_layout()
plt.savefig("charts/8_correlation.png", dpi=150); plt.close()

print("\nDone. Charts saved in /charts")
