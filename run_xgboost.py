# Pipeline 1: GNSS Spoofing/Jamming Detection using XGBoost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# 1. Load the dataset
data = pd.read_csv('/home/nader/Desktop/GNSS/gnss_dataset.csv')  # adjust filename as needed


# Separate features and labels
X = data.drop(columns=['label']).values
y = data['label'].values


 
# If scaling was desired (not strictly needed for XGBoost), you could use:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# 2. Split into training and testing sets with stratification to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Define and train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')            # multi-logloss for multi-class

model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate performance
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)

# Detailed classification report (precision, recall, F1 for each class)
target_names = ['Normal', 'Spoofing', 'Jamming']
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
