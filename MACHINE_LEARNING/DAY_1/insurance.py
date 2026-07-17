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

df_cleaned.rename(columns={
    'sex' :'is_female',
    'smoker': 'is_smoker'
                          },inplace = True)

df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)
df_cleaned = df_cleaned.astype(int)
# print(df_cleaned.head())

# FEATURE ENGINEERING AND EXTRATION

df_cleaned['bmi_cat'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
     labels=['underweight', 'normal', 'overweight', 'obese']
)

# print(df_cleaned.head())

df_cleaned = pd.get_dummies(df_cleaned,columns=['bmi_cat'],drop_first=True)
df_cleaned = df_cleaned.astype(int)
# print(df_cleaned.head())

from sklearn.preprocessing import StandardScaler
cols = ['age','bmi','children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

# print(df_cleaned.columns)

from scipy.stats import pearsonr
selected_features = ['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges',
       'region_northwest', 'region_southeast', 'region_southwest',
       'bmi_cat_normal', 'bmi_cat_overweight', 'bmi_cat_obese']

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
# print(correlation_df.sort_values(by='Pearson Correlation', ascending=False))

cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_cat_normal', 'bmi_cat_overweight', 'bmi_cat_obese'
]
from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)