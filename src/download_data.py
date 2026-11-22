# import kagglehub
# import shutil
# import os

# Download dataset
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# # Create data folder if it doesn't exist
# os.makedirs("data", exist_ok=True)

# # Copy CSV file into your project
# shutil.copy(
#     os.path.join(path, "creditcard.csv"),
#     "data/creditcard.csv"
# )

# Download latest version
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# print("Path to dataset files:", path)
#Login to Kaggle
# kagglehub.login()

# # Download dataset
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# # Create data folder
# os.makedirs("data", exist_ok=True)

# # Copy dataset
# source = os.path.join(path, "creditcard.csv")
# destination = os.path.join("data", "creditcard.csv")

# shutil.copy(source, destination)

# print("Dataset saved successfully!")
# kagglehub.login()

# # Download version 3 directly into data folder
# path = kagglehub.dataset_download(
#     "mlg-ulb/creditcardfraud/versions/3",
#     output_dir="data"
# )

# print("Dataset downloaded to:", path)
# kagglehub.login(validate_credentials=False)

# # Download dataset
# path = kagglehub.dataset_download(
#     "mlg-ulb/creditcardfraud/versions/3",
#     output_dir="data"
# )

# print("Dataset downloaded to:", path)
import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())