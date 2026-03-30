import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/bovine_health_dataset_2000_rows.csv")

data['disease'] = data['disease'].map({'healthy': 0, 'sick': 1})

X = data[['temperature', 'humidity', 'activity', 'heart_rate']]
y = data['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# Save best model
if rf_acc > lr_acc:
    best_model = rf
    best_acc = rf_acc
    model_name = "Random Forest"
else:
    best_model = lr
    best_acc = lr_acc
    model_name = "Logistic Regression"

joblib.dump(best_model, "model.pkl")

# Save accuracy
with open("accuracy.txt", "w") as f:
    f.write(f"{model_name} Accuracy: {best_acc}")

# KMeans
kmeans = KMeans(n_clusters=2)
data['cluster'] = kmeans.fit_predict(X)

# Save graph image
plt.figure()
plt.scatter(data['temperature'], data['heart_rate'], c=data['cluster'])
plt.xlabel("Temperature")
plt.ylabel("Heart Rate")
plt.title("Cow Behavior Clustering")

plt.savefig("static/graph.png")  # IMPORTANT
plt.close()

print("✅ Model + Graph + Accuracy saved")

