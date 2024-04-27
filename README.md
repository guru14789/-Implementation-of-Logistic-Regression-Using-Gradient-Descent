# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook

## Algorithm
1. Import the data file and import numpy, matplotlib, and scipy.
2. Visualize the data and define the sigmoid function, cost function, and gradient descent.
3. Plot the decision boundary.
4. Calculate the y-prediction. 

## Program:


Program to implement the Logistic Regression Using Gradient Descent.

Developed by: SREEKUMAR S

RegisterNumber:  212223240157

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
```
![Screenshot 2024-04-27 101738](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/3d3691f1-3988-49ac-9190-617f86176ea1)

```

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![Screenshot 2024-04-27 101746](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/15521dcf-6d64-4d19-83ac-0f60bf22ad5f)

```

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![Screenshot 2024-04-27 101755](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/29aa71cd-ac89-470a-a0de-91701f828b74)

```

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
```
![Screenshot 2024-04-27 101802](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/1e0b19b2-2eee-42d8-a627-d817c9b9bad8)

```

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
```
![Screenshot 2024-04-27 101808](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/66eeccfd-ce69-4dc3-b827-656beb2e857b)

```

print(y_pred)
print(y)
```
![Screenshot 2024-04-27 101815](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/31f8214d-2797-4af1-a0e2-a987825d5b11)

```

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![Screenshot 2024-04-27 101822](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/cf61d7c8-9698-4ec1-a104-f17ea6e7cfee)

```

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```
![Screenshot 2024-04-27 101828](https://github.com/guru14789/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151705853/3a8e83b9-8932-482a-a3ce-7c8b699c7276)


## Result:
Thus the program to implement the Logistic Regression Using Gradient Descent is written and verified using Python programming.

