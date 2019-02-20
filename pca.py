import pandas as pd 
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing, svm 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
url = "C:/Users/Madhur Kabra/Desktop/Python 3.6.5/madhur.csv"
df = pd.read_csv(url)

features = ['number','timestamp','weekend','holiday','temperature','semester','is_durin','month','hour']
x = df.loc[:,features].values

y = df.loc[:,['day']].values

#print(x)
#print(y)

x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principalComponents1','principalComponents2'])

finaldf = pd.concat([principalDf,df[['day']]],axis=1)
principalDf = ['principalComponents1','principalComponents2']
x1 = finaldf.loc[:,principalDf].values

y1 = finaldf.loc[:,['day']].values
#print(x1)
#print(y1)
y1=y1.ravel()
test_size = 0.33
#seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=test_size)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
accuracy = model.score(X_train,Y_train)
#print(accuracy)
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
#model = LogisticRegression()
#pca.fit(X, y)
#accuracy = pca.score(X,y)
print(accuracy)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

    







fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('principalComponents1',fontsize =15)
ax.set_ylabel('principalComponents2',fontsize =15)
ax.set_title('2 component PCA', fontsize = 20)
Healths = [4,5]
colors = ['r','g']
for day,color in zip(Healths,colors):
    indicesToKeep = finaldf['day'] == day
    ax.scatter(finaldf.loc[indicesToKeep, 'principalComponents1'],finaldf.loc[indicesToKeep,'principalComponents2'],c=color,s=50)
    ax.legend(Healths)
    ax.grid()

plt.show()