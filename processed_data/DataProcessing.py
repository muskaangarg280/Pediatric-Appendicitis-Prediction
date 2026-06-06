import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Copying the raw data into new csv file
# df = pd.read_csv("../raw_data/tabular_data.csv")
# df.to_csv("processed_data.csv", index = False)

#------------------------------------------
#
# # Removing unnamed columns and null columns
#
# df = pd.read_csv("processed_data.csv")
#
# # Starting by identifying no.of nulls in each column, and then categorizing them into null and empty columns
# null_count = df.isnull().sum()
# finding_null_columns = null_count[null_count > 300].index.tolist()
# finding_empty_columns = [col for col in df.columns if str(col).startswith("Unnamed")]
#
# # Combining both the lists of null and empty columns and then respectively dropping them from DataFrame.
# dropping_columns = list(set(finding_null_columns + finding_empty_columns))
# df = df.drop(columns=dropping_columns)
#
# # Using for loop to print out the names of the dropped columns along with no.of nulls they had, and then saving the cleaned DataFrame.
# print("Dropped columns:")
# for col in dropping_columns:
#     count = null_count.get(col, "NA (empty column)")
#     print(f"{col}: {count} nulls")
#
# df.to_csv("processed_data.csv", index=False)
#
#------------------------------------------

# df = pd.read_csv("processed_data.csv")
#
# #adds a new series to count number of missing values in each row
# df['null_count'] = df.isnull().sum(axis=1)
#
# #counts number of nulls in each row to sort
# nulls_per_row = df['null_count'].value_counts().sort_index()
#
# print("Number of rows vs number of nulls in each row:")
# print(nulls_per_row)

#------------------------------------------

# df = pd.read_csv("processed_data.csv")
#
# #adds a new series to count number of missing values in each row
# df['null_count'] = df.isnull().sum(axis=1)
# print(f"Rows before dropping: {len(df)}")
#
# #keeps only rows that have 4 or fewer missing values
# df = df[df['null_count'] <= 4]
#
# #removes temporary series as it's no longer needed
# df = df.drop(columns=['null_count'])
#
# print(f"Rows after dropping: {len(df)}")
# df.to_csv("processed_data.csv", index=False)

#------------------------------------------

# df = pd.read_csv("processed_data.csv")
#
# # Converting the gender categories to numeric labels using label encoder
# gender_col = "Sex"
# label_encoder = LabelEncoder()
# df[gender_col] = label_encoder.fit_transform(df[gender_col])
#
# # All the columns which has the values yes or no
# yes_no_cols = [
#    "Migratory_Pain",
#    "Lower_Right_Abd_Pain",
#    "Contralateral_Rebound_Tenderness",
#    "Coughing_Pain",
#    "Nausea",
#    "Loss_of_Appetite",
#    "Dysuria",
#    "Psoas_Sign",
#    "Ipsilateral_Rebound_Tenderness",
#    "Appendix_on_US",
#    "Neutrophilia",
#    "Free_Fluids",
#    "US_Performed"
# ]
# # for loop to convert all the yes values to 1 and no to 0
# for col in yes_no_cols:
#    df[col] = df[col].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == "yes" else 0)
#
# df.to_csv("processed_data.csv", index=False)

#------------------------------------------

# df = pd.read_csv("processed_data.csv")
#
# # Columns that contain image data
# image_cols = [
#    "Appendix_on_US",
#    "Appendix_Diameter",
#    "US_Performed",
#    "Free_Fluids"
# ]
#
# # Linking tabular and image datasets
# link_col = ["US_Number"]
#
# # To get the tabular columns that doesn't contain image-based data
# tabular_cols = [col for col in df.columns if col not in image_cols]
#
# # Splitting the original DataFrame into tabular and image-based datasets, each retaining the linking column
# df_tabular = df[tabular_cols]
# df_image = df[image_cols + link_col]
#
# df_tabular.to_csv("dataset_tabular.csv", index=False)
# df_image.to_csv("dataset_image.csv", index=False)

#------------------------------------------

# dataset_image = pd.read_csv("dataset_image.csv")
# dataset_tabular = pd.read_csv("dataset_tabular.csv")
#
# # Dictionary to rename old column names
# rename_image_cols = {
#     "US_Number": "Img_Ref_Num",
#     "Appendix_on_US": "Img_Appendix_Present",
#     "Appendix_Diameter": "Img_Appendix_Diameter",
#     "US_Performed": "Img_UltraSound_Performed",
#     "Free_Fluids": "Img_Free_Fluids"
# }
#
# # Renaming columns in image dataset
# dataset_image = dataset_image.rename(columns=rename_image_cols)
#
# # Renaming US_Number in tabular dataset
# dataset_tabular = dataset_tabular.rename(columns={"US_Number": "Img_Ref_Num"})
#
# dataset_tabular.to_csv("dataset_tabular.csv", index=False)
# dataset_image.to_csv("dataset_image.csv", index=False)

#------------------------------------------
#
# # Feature Correlation with the Target Features
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
#
# # Load data
# df = pd.read_csv("processed_data.csv")
#
# # Check data summary
# print(df.info())
# print(df.describe())
#
# # Define target variables
# targets = ["Diagnosis", "Severity", "Management"]
#
# # Encode categorical columns temporarily
# df_encoded = df.copy()
# le = LabelEncoder()
#
# for col in df_encoded.columns:
#     if df_encoded[col].dtype == "object":
#         df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#
# # Compute correlation matrix
# corr = df_encoded.corr()
#
# # Plot full heatmap
# plt.figure(figsize=(14,10))
# sns.heatmap(corr, cmap="coolwarm", center=0)
# plt.title("Correlation Heatmap (Including Encoded Categorical Features)")
# plt.show()
#
# # Target Specific Correlation Bar Chart
# for target in targets:
#     if target in corr.columns:
#         corr_target = corr[target].drop(target).sort_values(ascending=False).head(10)
#
#         plt.figure(figsize=(8, 4))
#         sns.barplot(
#             x=corr_target.values,
#             y=corr_target.index,
#             hue=corr_target.index,
#             palette="coolwarm",
#             legend=False
#         )
#         plt.title(f"Top Correlated Features with {target}")
#         plt.xlabel("Correlation Coefficient")
#         plt.ylabel("Feature")
#         plt.tight_layout()
#         plt.show()
#
# # Histograms of top correlated features with each target variable
# top_n = 1
#
# for target in targets:
#     if target in corr.columns:
#         # Get top correlated features (excluding the target itself)
#         corr_target = corr[target].drop(target).sort_values(ascending=False).head(top_n)
#         top_features = corr_target.index.tolist()
#
#         # Print top 10 features (regardless of top_n)
#         print(f"\nTop 10 features correlated with {target}:\n")
#         print(corr[target].drop(target).sort_values(ascending=False).head(10))
#
#         # Make histograms
#         print(f"Generating histograms for {target}...\n")
#         for feature in top_features:
#             plt.figure(figsize=(6,4))
#             sns.histplot(data=df, x=feature, hue=target, multiple="stack",
#                          palette="pastel", kde=True)
#             plt.title(f"Distribution of {feature} by {target}")
#             plt.xlabel(feature)
#             plt.ylabel("Count")
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             plt.show()
#




