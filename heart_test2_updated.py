import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb


heart = pd.read_csv("heart2.csv")
# splitting the dataset
x = heart.drop("output", axis=1)
y = heart["output"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# training the model

model = lgb.LGBMClassifier()
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],verbose=20,eval_metric='logloss')

print(cross_val_score(model, x, y, scoring="accuracy", cv = 10))
mean_score = cross_val_score(model, x, y, scoring="accuracy", cv = 10).mean()
std_score = cross_val_score(model, x, y, scoring="accuracy", cv = 10).std()
print(mean_score)
print(std_score)

# giving inputs to the machine learning model
# features = [[sepal_length, sepal_width, petal_length, petal_width]]
features = np.array([[63,1,3,145,233,1,0,150,0,2.1,0,0,1],[58,0,1,136,319,1,0,152,0,0,2,2,2]])
# using inputs to predict the output
prediction = model.predict(features)
print("Prediction: {}".format(prediction))