import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
print(x_train)

print(x_test)

print(y_train)

print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)

print(x_test)

from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0)
cls.fit(x_train, y_train)

y_pred = cls.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(x_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from matplotlib.colors import ListedColormap
x_set,y_set = sc.inverse_transform(x_train,y_train)
x1,x2 = np.meshgrid(np.arrange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.25),
                    np.arrange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.25))
plt.contourf(x1,x2,cls.predict(sc.transform(np.array[x1.ravel(),x2.ravel()].T)).reshape(x1.shape), alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=i)
plt.title('Logistic Regression (Training set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
x_set,y_set = sc.inverse_transform(x_test,y_test)
x1,x2 = np.meshgrid(np.arrange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.25),
                    np.arrange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.25))
plt.contourf(x1,x2,cls.predict(sc.transform(np.array[x1.ravel(),x2.ravel()].T)).reshape(x1.shape), alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=i)
plt.title('Logistic Regression (Testing set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()