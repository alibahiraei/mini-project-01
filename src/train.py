from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

#load dataset
BAS_DIR= Path(__file__).resolve().parent.parent
DATA_PATH= BAS_DIR / "data" / "creditcard.csv"

df=pd.read_csv(DATA_PATH)

X=df.drop("Class",axis=1)
y=df["Class"]

X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.2,stratify=y,
    random_state=42
)

#creat logistic regression pipline
logistic_model=Pipeline([
    ("scaler", StandardScaler()),
    ("model",LogisticRegression(max_iter=1000))
])

# train the model
logistic_model.fit(X_train,y_train)
#print("Logistic Regression training completed.")

# Create KNN Pipeline

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

# Train the model
knn_model.fit(X_train, y_train)

#print("KNN training completed.")

# Create Decision Tree Pipeline

tree_model = Pipeline([
    ("model", DecisionTreeClassifier(random_state=42))
])

# Train the model
tree_model.fit(X_train, y_train)

#print("Decision Tree training completed.")

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    print(f"\n===== {model_name} =====")

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,zero_division=0))

evaluate_model(
    logistic_model, X_test, y_test, "Logistic Regression"
)

evaluate_model(
    knn_model, X_test, y_test, "KNN"
)

evaluate_model(
    tree_model, X_test, y_test, "Decision Tree"
)