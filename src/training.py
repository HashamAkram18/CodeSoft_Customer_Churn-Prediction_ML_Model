import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
import pandas as pd

# Get the directory of the current script
script_directory = os.path.dirname(__file__)

# Define the directory where the processed data is stored
processed_data_dir = os.path.join(script_directory, '..', 'artifacts', 'processed_data')

# Load preprocessed training and testing data
train_file_path = os.path.join(processed_data_dir, 'train.csv')
test_file_path = os.path.join(processed_data_dir, 'test.csv')
train_labels_file_path = os.path.join(processed_data_dir, 'train_labels.csv')
test_labels_file_path = os.path.join(processed_data_dir, 'test_labels.csv')

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Load train and test labels
train_labels = pd.read_csv(train_labels_file_path)['Exited']
test_labels = pd.read_csv(test_labels_file_path)['Exited']

# Instantiate the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(train_data, train_labels)

# Predict on the test data
y_pred = rf_classifier.predict(test_data)

# Calculate accuracy
accuracy = accuracy_score(test_labels, y_pred)
print("Accuracy:", accuracy)

# Calculate R2 score
r2 = r2_score(test_labels, y_pred)
print("R2 Score:", r2)

# Save the trained model
model_dir = os.path.join(script_directory, 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'classifier.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf_classifier, f)

# Save the metrics
metrics_dir = os.path.join(script_directory, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)

# Accuracy
accuracy_path = os.path.join(metrics_dir, 'accuracy.txt')
with open(accuracy_path, 'w') as f:
    f.write(str(accuracy))

# R2 Score
r2_path = os.path.join(metrics_dir, 'r2_score.txt')
with open(r2_path, 'w') as f:
    f.write(str(r2))

print("Model saved to:", model_path)
print("Metrics saved to:", metrics_dir)
