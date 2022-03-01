import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
heart = pd.read_csv("heart.csv")
# splitting the dataset
x = heart.drop("output", axis=1)
y = heart["output"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# training the model
from sklearn.naive_bayes import GaussianNB
knn = GaussianNB()
knn.fit(x_train, y_train)

print(cross_val_score(knn, x, y, scoring="accuracy", cv = 10))
mean_score = cross_val_score(knn, x, y, scoring="accuracy", cv = 10).mean()
std_score = cross_val_score(knn, x, y, scoring="accuracy", cv = 10).std()
print(mean_score)
print(std_score)

# giving inputs to the machine learning model
# features = [[sepal_length, sepal_width, petal_length, petal_width]]
features = np.array([[63,1,3,145,233,1,0,150,0,2.1,0,0,1],[58,0,1,136,319,1,0,152,0,0,2,2,2]])
# using inputs to predict the output
prediction = knn.predict(features)
print("Prediction: {}".format(prediction))