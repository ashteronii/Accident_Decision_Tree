import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Load the cleaned Road Accident Survival dataset.
accident = pd.read_csv("Accident_Cleaned.csv")

# Extract the target variable ('Survived') and remaining features.
y = accident['Survived'].copy().to_numpy()
X = accident.drop(columns=['Survived']).copy().to_numpy()

# Split the data int training and validation sets (70%/30% split).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize a Decision Tree Classifier.
clf = DecisionTreeClassifier()

# Perform grid search with 5-fold cross-validation to find optimal max_depth between 2 and 15.
parameters = {"max_depth": range(2,16)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)

# Output the best hyperparameters and cross-validation score.
print("Decision Tree Best Params:", grid_search.best_params_)
print("Decision Tree Best Score:", grid_search.best_score_)

# Display grid search results.
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_max_depth','mean_test_score','rank_test_score']])

# Train a new model using the optimal max_depth.
best_max_depth = grid_search.best_params_['max_depth']
best_clf = DecisionTreeClassifier(max_depth=best_max_depth)
best_clf.fit(X_train, y_train)

# Evaluate the new model's performance on training and testing data to check for overfitting.
print(f"Optimal value for max_depth: {best_max_depth}")
print(f"Train Score: {best_clf.score(X_train, y_train):.3f}")
print(f"Test Score: {best_clf.score(X_test, y_test):.3f}")

# Visualize the top 3 levels of the decision tree for interpretability.
best_tree = grid_search.best_estimator_
plt.figure(figsize=(12, 8))
plot_tree(best_tree,
          max_depth=3,
          filled=True,
          feature_names=['Age','Gender','Speed_of_Impact','Helmet_Used','Seatbelt_Used'],
          class_names=[str(cls) for cls in best_clf.classes_]
          )
plt.show()
