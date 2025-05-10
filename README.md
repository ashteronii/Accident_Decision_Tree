# Road Accident Survival Predicition using Decision Trees

This project applies a Decision Tree Classifier to predict whether an individual survives a road accident based on demographic and situational factors such as age, gender, and the use of safety equipment. The project includes data cleaning, model tuning with cross-validation, and tree visualization of the top levels of the resulting decision tree to aid in interpretability.

---

## Dataset

- Himel Sardeu (2025). *Road Accident Survival Dataset*. Kaggle. https://www.kaggle.com/datasets/himelsarder/road-accident-survival-dataset
- This dataset contains accident records and related personal and safety characteristics.
- Target variable: `Survived` (0 = Did not survive, 1 = Survived)
- Input features include a mix of demographic and situational data relevant to accident outcomes.

---

## Features

The following variables were used in model training:
- `Age`: The age of the individual in the accident.
- `Gender`: The gender of the individual (converted to binary: Male = 0, Female = 1).
- `Speed_of_Impact`:  Speed at which the crash occurred.
- `Helmet_Used`: Indicator variable for helmet usage (No = 0, Yes = 1).
- `Seatbelt_Used`: Indicator variable for seatbelt usage (No = 0, Yes = 1).
- `Survived`: Target variable indicating survival (0 = survived, 1 = did not survive).

---

## Project Structure

- `Accident_Cleaned.csv`: Cleaned dataset used for training.
- `Accident_Cleaned.py`: Data cleaning script.
- `Accident_Decision_Tree.py`:  Model training and evaluation script.
- `accident.csv`: Raw dataset from Kaggle.
- `README.md`: This file.

---

## Model Overview

- Model: `DecisionTreeClassifier` from `sklearn.tree`
- Hyperparameter Tuning: `GridSearchCSV` was used to fin the best `max_depth` between 2 and 15.
- Cross-validation: 5-fold Cross-Validation was used to avoid overfitting.

After selecting the best `max_depth`, the final model was trained and evaluated on both training and test data. The top 3 levels of the tree were visualized to improve interpretability.

---

## Class Imbalance Consideration

To determine whether class weighting was necessary, the target variable distribution was checked:
```bash
print(accident['Survived'].value_counts())
```
The resulting output was:
- Class 0 (survived): 100 instances.
- Class 1 (did not survive): 96 instances.

Since the class distribution is relatively balanced (100 vs. 96), it was not necessary to use `class_weight='balanced'`.

---

## Results

Due to the randomness of `train_test_split`, the optimal `max_depth`, training accuracy, testing accuracy, and resulting decision tree varies between each run. 

The decision tree was visualized to a depth of 3 for interpretability and to observe the most influential splitting rules based on the selected features.

---

## How to Run

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/ashteronii/Accident_Decision_Tree.git
   cd Accident_Decision_Tree
   ```
2. Install required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
3. Run the cleaning script:
   ```bash
   python Accident_Cleaned.py
   ```
4. Run the main script:
   ```bash
   python Accident_Decision_Tree.py
   ```
   
---

## Dependencies

- Python 3.8+
- pandas
- scikit-learn
- matplotlib

