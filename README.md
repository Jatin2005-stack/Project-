pip install pandas numpy scikit-learn matplotlib
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Scores': [10,20,30,40,50,60,70,80,90,100]
}

df = pd.DataFrame(data)

# Split data into X (input) and y (output)
X = df[['Hours']]
y = df['Scores']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
predicted = model.predict([[7]])
print("Predicted Score for 7 hours:", predicted[0])

# Plot graph
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Score")
plt.show()

