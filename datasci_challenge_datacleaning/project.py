import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading and exploration
# 1. Loading the dataset

df = pd.read_csv("data.csv")
df.shape
df.head(5)

# 2. feature description

print(df.describe())

# the dataset represents a variety of different numerical parameters with their respective mean, its standard error and its worst
# along with just one categorical column describing whether the case is benign or malignant

len(df.columns)

# 3. data types

df.dtypes

for col in df.columns:
    if df[col].nunique() < 10:
        print(col, df[col].nunique())


# 4. Column names

df.columns

df = df.rename(columns={
    "concave points_mean": "concave_points_mean",
    "concave points_se": "concave_points_se",
    "concave points_worst": "concave_points_worst"
})


# the dataset represents a variety of different numerical parameters with their respective mean, its standard error and its worst
# along with just one categorical column describing whether the case is benign or malignant

# 5.Missing values 

df.isnull().sum()

for col in df.columns:
   missing = (df[col].isnull().sum() / df.shape[0]) * 100
   print(f"{col}: {missing:.2f}%")

df = df.drop("Unnamed: 32", axis=1)

# 6.Indexion and delection

df_malignant = df[df["diagnosis"] == "M"]
df_5_ligne = df.loc[0:5, "diagnosis"]
df_ilo = df.iloc[0:5]

# 7.descriptive statistics

df.describe()

# 8.grouping and aggregation

df.groupby("diagnosis").mean()
df.groupby("diagnosis").median()
df.groupby("diagnosis").count()

# there does not seen to be any extreme value although you would need a proffessional to give help make a decision 

# Direct Questions

# Q1
df["diagnosis"].value_counts().idxmax()
# Q2
print(df["area_se"].dtype)
# Q3
diff = df.groupby("diagnosis")["area_mean"].mean()
abs(diff["B"]-diff["M"])
# Q4
df.drop(["diagnosis", "id"], axis=1).std().idxmin()
# Q5
number_of_worst_peri=len(df[df["perimeter_worst"]>100])

# visualisation

# distribution of features

cols_num = df.drop(["id", "diagnosis"], axis=1).columns

n_cols = 4
n_rows = (len(cols_num) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))

axes = axes.flatten()

for i, col in enumerate(cols_num):
    sns.histplot(data=df, x=col, hue="diagnosis", ax=axes[i])

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])   

plt.show()

# pairwise relationship 

sns.pairplot(df, palette="viridis")
plt.show()

# relationship between diagnosis and specific features

for i, col in enumerate(cols_num):
    sns.violinplot(data=df, x=col, hue="diagnosis", ax=axes[i])

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])   

plt.show()

# subsetting and caltegorical features

col_three = ["smoothness_mean", "smoothness_se", "smoothness_worst"]

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

df["perimeter_bin"] = pd.qcut(df["perimeter_mean"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

for i, col in enumerate(col_three):
    sns.boxplot(data=df, x="perimeter_bin", y=col, hue="diagnosis", ax=axes[i])

sns.pointplot(data=df, x="diagnosis", y="radius_mean")

plt.show()







