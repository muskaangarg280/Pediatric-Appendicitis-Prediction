# Exploratory Data Analysis (EDA)
# Dataset: dataset_tabular.csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Load dataset
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT.parent / "dataset_tabular.csv")

# Check data summary -----
print(df.info())
print(df.describe())

# Define target variables
targets = ["Diagnosis", "Severity", "Management"]

# Encode categorical columns temporarily
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Compute correlation matrix -----
corr = df_encoded.corr()

# Plot full heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Including Encoded Categorical Features)")
plt.show()

# Top correlated features with each target variable -----
top_n = 1

for target in targets:
    if target in corr.columns:
        # Get top correlated features (excluding the target itself)
        corr_target = corr[target].drop(target).sort_values(ascending=False).head(top_n)
        top_features = corr_target.index.tolist()

        # Print top features (regardless of top_n)
        print(f"\nTop features correlated with {target}:\n")
        print(corr[target].drop(target).sort_values(ascending=False).head(3))

# Histogram of Entire Dataset -----
# Exclude columns
exclude_cols = ["Img_Ref_Num", "Management", "Severity", "Diagnosis_Presumptive", "Diagnosis"]
df_filtered = df.drop(columns=exclude_cols, errors='ignore')

#Categorical columns to numeric
df_filtered['Stool_numeric'] = df['Stool'].map({'normal': 0, 'constipation': 1, 'diarrhea': 2})
df_filtered['Peritonitis_numeric'] = df['Peritonitis'].map({'no': 0, 'local': 1, 'generalized': 2})

#Automatically generate a distinct color for each column
num_cols = len(df_filtered.columns)
cmap = plt.get_cmap('gist_ncar')
colors = [cmap(i / num_cols) for i in range(num_cols)]

# Plotting the histogram
df_filtered.plot(kind='hist',alpha=0.5, bins=26,figsize=(12, 8),edgecolor='black',subplots=False,color=colors)
plt.title("Histogram for Dataset")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend(df_filtered.columns, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()

# Alvarado Score vs Diagnosis -----
plt.figure(figsize=(8,6))
sns.boxplot(
    data=df,
    x="Diagnosis",
    y="Alvarado_Score",
    palette="Set2"
)
sns.stripplot(
    data=df,
    x="Diagnosis",
    y="Alvarado_Score",
    color="black",
    alpha=0.3
)
plt.title("Alvarado Score vs Appendicitis Diagnosis", fontsize=14)
plt.xlabel("Diagnosis")
plt.ylabel("Alvarado Score")
plt.tight_layout()
plt.show()


# Appendix Diameter vs Management (split by Severity) -----
# Filter required rows (drop missing Appendix_Diameter, Management, Severity)
plot_df = df.dropna(subset=["WBC_Count", "Management", "Severity"])

# Convert to categorical for cleaner plots
plot_df["Management"] = plot_df["Management"].astype("category")
plot_df["Severity"] = plot_df["Severity"].astype("category")

# Set style
sns.set_theme(style="whitegrid")

# Faceted boxplots
g = sns.catplot(
    data=plot_df,
    x="WBC_Count",
    y="Management",
    hue="Management",
    kind="box",
    col="Severity",
    palette="Set2",
    height=5,
    aspect=1
)

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("WBC_Count vs Appendicitis Type of Management")
g.set_axis_labels("WBC_Count", "Management")
plt.show()