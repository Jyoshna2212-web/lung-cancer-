

Original file is located at
    https://colab.research.google.com/drive/19Pg_57p1G3Z4nqZGk42SI8fx2BgMz9y9
"""

from google.colab import files

# Upload the titanic.csv file
uploaded = files.upload()

# Lung Cancer Prediction Project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("lungcancer.csv")

# Encode categorical variables
le = LabelEncoder()
df["GENDER"] = le.fit_transform(df["GENDER"])   # M/F -> 0/1
df["LUNG_CANCER"] = le.fit_transform(df["LUNG_CANCER"])  # NO/YES -> 0/1

# Basic info
print("Dataset Shape:", df.shape)
print(df.head())

# EDA - countplot for target
sns.countplot(x="LUNG_CANCER", data=df, palette="Set2")
plt.title("Distribution of Lung Cancer Cases")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Split dataset
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Cancer", "Cancer"],
            yticklabels=["No Cancer", "Cancer"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
