import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('CSV//insurance.csv')
# print(df)

# EDA
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.columns)

num_col = ['age', 'bmi', 'children', 'charges']

# for col in num_col:
#     # plt.figure(figsize=(6,4))
#     sns.histplot(df[col], kde=True, bins=20)
#     plt.show()

# sns.countplot(x=df['sex'])
# plt.show()
# sns.countplot(x=df['children'])
# plt.show()
# sns.countplot(x=df['smoker'])
# plt.show()
# sns.countplot(x=df['region'])
# plt.show()

# for col in num_col:
#     plt.figure(figsize=(10,8))
#     sns.boxplot(x=df[col])
#     plt.show()

# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(numeric_only=num_col),annot=True)
# plt.show()

# DATA CLEANING AND PROCESSING

df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)
df_cleaned['sex'] = df_cleaned['sex'].map({"male" : 0, "female" : 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({'no' :0, 'yes' :1})
df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)
df_cleaned = df_cleaned.astype(int)
# print(df_cleaned.head())

# FEATURE ENGINEERING AND EXTRATION

df_cleaned['bmi_cat'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
     labels=['underweight', 'normal', 'overweight', 'obese']
)

print(df_cleaned.head())