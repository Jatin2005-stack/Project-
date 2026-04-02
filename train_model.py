import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/bovine_health_dataset_2000_rows.csv")

# Features
X = df[['temperature', 'humidity', 'activity', 'heart_rate']]
y = df['disease'].map({'healthy': 0, 'sick': 1})

# ✅ STRATIFY FIX
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# ✅ IMPROVED MODEL
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(rf, "model.pkl")

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
joblib.dump(kmeans, "kmeans.pkl")

# Save accuracy
with open("accuracy.txt", "w") as f:
    f.write(f"RF Accuracy: {acc}")

# Accuracy graph
plt.figure()
plt.bar(['RF'], [acc])
plt.savefig("static/accuracy_graph.png")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig("static/confusion_matrix.png")
plt.close()

print("Training Fixed ✅")