from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
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

# evaluate_model(
#     logistic_model, X_test, y_test, "Logistic Regression"
# )

# evaluate_model(
#     knn_model, X_test, y_test, "KNN"
# )

# evaluate_model(
#     tree_model, X_test, y_test, "Decision Tree")


# 5-Fold Stratified Cross Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "Decision Tree": tree_model
}

# for name, model in models.items():

#     scores = cross_validate(
#         model,
#         X_train,
#         y_train,
#         cv=cv,
#         scoring=scoring
#     )

#     print(f"\n===== {name} =====")
#     print("Mean Precision:", scores["test_precision"].mean())
#     print("Mean Recall:", scores["test_recall"].mean())
#     print("Mean F1:", scores["test_f1"].mean())

# Experiment 1: Effect of Scaling on KNN

knn_without_scaling = KNeighborsClassifier(n_neighbors=5)

knn_with_scaling = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

experiments = {
    "Without Scaling": knn_without_scaling,
    "With Scaling": knn_with_scaling
}

# for name, model in experiments.items():

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     print(f"\n===== KNN {name} =====")
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1-score:", f1_score(y_test, y_pred))

#Option 2: Decision Tree hyperparametr
# for i in [2,5,10,None]:
#     tree_model_deep=DecisionTreeClassifier(max_depth=i,random_state=42)
#     tree_model_deep.fit(X_train,y_train)
#     y_pred_deep=tree_model_deep.predict(X_test)
#     print(f"\n===== Parameter Value {i} =====")
#     print("Precision:", precision_score(y_test, y_pred_deep))
#     print("Recall:", recall_score(y_test, y_pred_deep))
#     print("F1-score:", f1_score(y_test, y_pred_deep))

# impact classification threshold
models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "Decision Tree": tree_model
}
Threshold=[0.3,0.5,0.7]

for model_name,model in models.items():
    y_proba=model.predict_proba(X_test)[:,1]
    print(f"\n===== {model_name} =====")
    for threshold in Threshold:
        y_pred=(y_proba>=threshold).astype(int)
        precision=precision_score(y_test,y_pred,zero_division=0)
        pricision=precision_score(y_test,y_pred,zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tp, fp, fn, tn = confusion_matrix(
            y_test, y_pred, labels=[0, 1]).ravel()

        print(f"\nThreshold: {threshold}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")