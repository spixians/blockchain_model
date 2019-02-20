import pandas
import numpy as np
from numpy import nan

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
url = "C:/Users/Madhur Kabra/Desktop/Python 3.6.5/madhur.csv"
df = pandas.read_csv(url)
#df = df.reset_index()
features = ['number','timestamp','weekend','holiday','temperature','semester','is_durin','month','hour']
X = df.loc[:,features].values
X.fillna(X.mean())
Y = df.loc[:,['day']].values
#y = np.array(Y)
#X.fillna(X.mean())
Y=Y.ravel()
test_size = 0.3
#seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size)
#np.nan_to_num(X_test)
# Fit the model on 33%4

model = LogisticRegression()
model.fit(X_train, Y_train)
accuracy = model.score(X_test,Y_test)
print(accuracy)
#X, y = make_blobs(n_samples=100, centers=2, n_features=9, random_state=1)
# fit final model
#model = LogisticRegression()
#model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=9, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
