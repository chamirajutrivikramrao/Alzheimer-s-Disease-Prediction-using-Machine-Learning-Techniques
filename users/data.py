import os
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Path to your dataset
csv_file_path = r'C:\Users\lenovo\Desktop\nandini\Alzheimers_Disease_Prediction\balanced_data.csv'

# Load the dataset
data = pd.read_csv(csv_file_path)

# Assuming the target column is named 'target'
# Replace 'target' with the actual column name in your dataset
target_column = 'target_column'

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Check the class distribution before balancing
print("Class distribution before balancing:")
print(y.value_counts())

# Use SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and target into a DataFrame
balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])], axis=1)

# Check the class distribution after balancing
print("Class distribution after balancing:")
print(balanced_data[target_column].value_counts())

# Save the balanced dataset to the same location
balanced_csv_path = r'C:\Users\lenovo\Desktop\nandini\Alzheimers_Disease_Prediction\balanced_data.csv'
balanced_data.to_csv(balanced_csv_path, index=False)

print(f"Balanced dataset saved to {balanced_csv_path}")
