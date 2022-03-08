import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import sys

import sys
np.set_printoptions(threshold=sys.maxsize)

import warnings
warnings.filterwarnings("ignore")

def prediction(prediction1,prediction2,prediction3,prediction4,prediction5):
    zero = [0 for i in range(len(prediction1))]
    one = [0 for i in range(len(prediction1))]
    predict = [0 for i in range(len(prediction1))]
    for i in range(len(prediction1)):
        if prediction1[i] == 0:
            zero[i] += 2
        else:
            one[i] += 2

        if prediction2[i] == 0:
            zero[i] += 1
        else:
            one[i] += 1

        if prediction3[i] == 0:
            zero[i] += 1
        else:
            one[i] += 1

        if prediction4[i] == 0:
            zero[i] += 1
        else:
            one[i] += 1

        if prediction5[i] == 0:
            zero[i] += 1
        else:
            one[i] += 1

        if one[i] > zero[i]:
            predict[i] = 1
        else:
            predict[i] = 0
    return predict


#importing the dataset
heart = pd.read_csv("heart2.csv")
# splitting the dataset
x = heart.drop("output", axis=1)
y = heart["output"]
heart = pd.read_csv("heart.csv")
a = heart.drop("output", axis=1)
b = heart["output"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# training the model


model = lgb.LGBMClassifier()
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],verbose=101,eval_metric='logloss')


#GaussianNB
gaussian = GaussianNB()  #80.5
gaussian.fit(x_train, y_train)


#RFC
rfc = RFC() #82.49
rfc.fit(x_train, y_train)


lr = LogisticRegression(random_state=0) #83.16
lr.fit(x_train, y_train)


#lda
LDA = lda()  #82.16
LDA.fit(x_train, y_train)




features = np.array(a)
# using inputs to predict the output
prediction1 = model.predict(features)
#print(prediction1)
prediction2 = gaussian.predict(features)
for i in range(len(prediction2)):
    if prediction2[i] == 1:
        prediction2[i] = 0
    else:
        prediction2[i] = 1
#print(prediction2)
prediction3 = rfc.predict(features)
#print(prediction3)
prediction4 = lr.predict(features)
#print(prediction4)
prediction5 = LDA.predict(features)
#print(prediction5)
prediction = prediction(prediction1,prediction2,prediction3,prediction4,prediction5)
#print(prediction,"\n\n") 
for i in range(len(prediction)):
    if prediction[i] == b[i]:
        prediction[i] = 1
    else:
        prediction[i] = 0
        print(i+2)
print(prediction)

#a = np.count(prediction,1)
#print(a)
#print(len(prediction))
unique, counts = np.unique(prediction, return_counts=True)

print(counts[0], counts[1])
print("accuracy", (100*counts[1])/len(prediction))