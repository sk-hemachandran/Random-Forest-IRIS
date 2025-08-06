# Random-Forest-IRIS
🌳 Random Forest on the Iris Dataset
📌 Overview:-
This project demonstrates the use of the Random Forest classification algorithm on the Iris dataset, one of the most well-known datasets in machine learning. The goal is to classify iris flowers into one of three species — Setosa, Versicolor, or Virginica — based on four features:

#Sepal length

#Sepal width

#Petal length

#Petal width

🤖 What is Random Forest?
Random Forest is an ensemble learning method used for classification and regression. It operates by building multiple decision trees during training time and outputting the mode of the classes (for classification) or mean prediction (for regression) of the individual trees.

🔑 Key Features of Random Forest:-
Combines the predictions of multiple decision trees to improve accuracy and reduce overfitting.

Introduces randomness by selecting random subsets of features and data samples for each tree.

Robust to noise and effective even with missing data or unbalanced datasets.

📊 About the Iris Dataset:-
The Iris dataset contains 150 samples of iris flowers, each with 4 numerical features and a target label representing one of the three species. It's widely used for testing classification algorithms due to its simplicity and interpretability.

🧠 Methodology:-
Data Loading: The dataset is loaded using scikit-learn's built-in Iris dataset.

Data Splitting: The data is split into training and testing sets (typically 70/30 or 80/20 split).

Model Building: A RandomForestClassifier from sklearn.ensemble is instantiated with specified parameters (e.g., n_estimators=100).

Training: The model is trained on the training data using the .fit() method.

Prediction: Predictions are made on the test data using the .predict() method.

Evaluation: The model's performance is evaluated using accuracy score, confusion matrix, and classification report.

✅ Results
The Random Forest classifier usually achieves high accuracy (~95-100%) on the Iris dataset due to its simplicity and clearly separated classes.

📈 Sample Code Snippet

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

🧾 Conclusion
This project demonstrates how Random Forest can be effectively used for multi-class classification tasks like the Iris dataset. It’s a great starting point for beginners to understand ensemble models and basic machine learning pipelines.

