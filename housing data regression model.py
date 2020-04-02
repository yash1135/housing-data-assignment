import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def hypothesis(theta,x):
    y_=theta[0]+theta[1]*x
    return y_

def gradient(x,y,theta):
    m=x.shape[0]
    grad=np.zeros((2,))
    for i in range(m):
        X=x[i]
        y_=hypothesis(theta,X)
        Y=y[i]
        grad[0]+=y_-Y
        grad[1]+=(y_-Y)*X
    return grad/m
    
def gradient_descent(x,y,max_steps=100,learning_rate=0.1):
    error_list=[]
    theta=np.zeros((2,))
    for i in range(max_steps):
        grad=gradient(x,y,theta)
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
        e=error(x,y,theta)
        error_list.append(e)
    return theta,error_list

def error(x,y,theta):
    m=x.shape[0]
    total_error=0.0
    for i in range(m):
        y_=hypothesis(theta,x[i])
        total_error+=(y_-y[i])**2
    return total_error/m

def r2_score(y,y_):
    num=np.sum((y-y_)**2)
    denum=np.sum((y-y.mean())**2)
    score=(1-num/denum)
    return score*100

df=pd.read_csv("training_data_housing_price_x1.csv")
df.head()
x=df[['x']]
y=df[['y']]
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)

plt.style.use('seaborn')
plt.scatter(X_train,Y_train)
plt.show()
X_train=X_train.values
Y_train=Y_train.values
x_train=X_train
u=X_train.mean()
std=X_train.std()
X_train=(X_train-u)/std
theta,error_list=gradient_descent(X_train,Y_train)
y_=hypothesis(theta,X_train)
plt.scatter(X_train,Y_train)
plt.plot(X_train,y_,color='orange',label='prediction')
plt.legend()
plt.show()

X_test=X_test.values
Y_test=Y_test.values
u=X_test.mean()
std=X_test.std()
X_test=(X_test-u)/std
y_pred=hypothesis(theta,X_test)
plt.scatter(X_test,Y_test)
plt.show()

plt.scatter(X_test,y_pred)
plt.show()

a=r2_score(Y_test,y_pred)
print(a)
