#importing some required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
iris=sns.load_dataset("iris")
print(iris.head())
print(iris.info())
print(iris.describe())

#data visuals
sns.countplot(x="species",data=iris)
plt.show()
sns.pairplot(iris,hue="species")
plt.show()
setosa=iris[iris["species"]=="setosa"]
sns.kdeplot(setosa["sepal_length"],setosa["sepal_width"],cmap="plasma",shade=True,shade_lowest=False)
plt.show()

#splitting training and testing data
x=iris.drop("species",axis=1)
y=iris["species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


#SVM without grid
s=SVC()
s.fit(x_train,y_train)
pred=s.predict(x_test)
print(" At times SVM model without Grid search proves good for small datasets.")
print("Confusion matrix :")
print(confusion_matrix(y_test,pred))
print("Classification report :")
print(classification_report(y_test,pred))

#grid
param_grid={"C":[0.1,1,10,100,1000],"gamma":[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3,refit=True)
grid.fit(x_train,y_train)
pred=grid.predict(x_test)
print("Confusion matrix :")
print(confusion_matrix(y_test,pred))
print("Classification report :")
print(classification_report(y_test,pred))