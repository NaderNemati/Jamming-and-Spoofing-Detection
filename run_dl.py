# Pipeline 2: GNSS Spoofing/Jamming Detection using a CNN on 2D Feature Images

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


data = pd.read_csv('/home/nader/Desktop/GNSS/gnss_dataset.csv')
X = data.drop(columns=['label']).values
y = data['label'].values


# These should be set based on domain knowledge of the simulation (number of satellites, time points, features).
# For example, assume 3 feature types (C/N0, Doppler, Power), 10 satellites, 60 time points:


n_sats = 8           # Number of satellites per sample
n_features = 3       # CNR, Doppler, SignalPower
n_timesteps = 30



# Verify that total feature count matches n_features * n_sats * n_timesteps
expected_cols = n_features * n_sats * n_timesteps
if X.shape[1] != expected_cols:
    raise ValueError(f"Feature count {X.shape[1]} does not match expected dimensions ({expected_cols}). Adjust n_sats/n_timesteps.")

# Each sample must be reshaped into [n_features, n_sats, n_timesteps] (or [n_sats, n_timesteps, n_features]) for imaging.
# We'll reshape to (n_sats, n_timesteps, n_features) which is a height x width x channels format.
X_images = X.reshape(-1, n_sats, n_timesteps, n_features)


X_images = X_images.astype('float32')

# Normalize the data to [0, 1] range
X_min = X_images.min()
X_max = X_images.max()
if X_max > X_min:
    X_images = (X_images - X_min) / (X_max - X_min)

# 2. Split into training and testing sets with stratification to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y, test_size=0.3, random_state=42, stratify=y)

# Convert labels to categorical (one-hot) for training if using categorical_crossentropy
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)


# 3. Define the CNN model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same',
                 input_shape=(n_sats, n_timesteps, n_features)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # dropout for regularization
model.add(Dense(3, activation='softmax')) 



model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

# 4. Train the model
model.fit(X_train, y_train_cat, epochs=200, batch_size=64, validation_split=0.1, verbose=2)

# 5. Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 6. Evaluate performance
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)

print("Classification Report:")
target_names = ['Normal', 'Spoofing', 'Jamming']
print(classification_report(y_test, y_pred, target_names=target_names))
