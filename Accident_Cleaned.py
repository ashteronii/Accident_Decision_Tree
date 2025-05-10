import pandas as pd

# Load the Road Accident Survival dataset from the specified CSV file.
# This dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/himelsarder/road-accident-survival-dataset
accident = pd.read_csv("accident.csv")

# Cleans and preprocesses the accident dataset.
# Steps:
#    - Drop missing values.
#    - Remove duplicate rows.
#    - Convert binary categorical features ('Gender', 'Helmet_Used', 'Seatbelt_Used') to 0/1.
# This ensures that we only work with complete and unique records for analysis.
accident = accident.dropna().drop_duplicates()
accident['Gender'] = accident['Gender'].map({'Male': 0, 'Female': 1})
accident['Helmet_Used'] = accident['Helmet_Used'].map({'No': 0, 'Yes': 1})
accident['Seatbelt_Used'] = accident['Seatbelt_Used'].map({'No': 0, 'Yes': 1})

# Save the cleaned version of the dataset to a new CSV file called 'Accident_Cleaned.csv'.
# This cleaned dataset will be used for training and further analysis.
accident.to_csv("Accident_Cleaned.csv", index=False)