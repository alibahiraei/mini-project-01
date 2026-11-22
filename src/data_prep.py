from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

#Anahysze data stracthure
# print(df.head())
# print(df.info())

#calculate discriptive statistics
# print(df.describe())

#handle missing values
#print("Total Missing Values:", df.isnull().sum().sum())

# Analyze class distribution
# print(df["Class"].value_counts())

# print("\nPercentage:")
# print(df["Class"].value_counts(normalize=True) * 100)


# Separate Features and Target
X=df.drop("Class",axis=1)
y=df["Class"]

# print("X shape",X.shape)
# print("y shape",y.shape)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.02,stratify=y,
    random_state=42
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True) * 100)
