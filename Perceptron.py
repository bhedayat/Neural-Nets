
# coding: utf-8

# In[22]:

#Flower Classifier with Perceptron
#Read in data

data = [] #training data
data1 = [] #test data
import numpy as np
import matplotlib.pyplot as plt
from random import randint

for line in file:
    l = (line.split(","))
    l[0] = float(l[0])
    l[1] = float(l[1])
    l[2] = float(l[2])
    l[3] = float(l[3])
    data.append (l)
for line1 in file1:
    h = (line1.split(","))
    h[0] = float(h[0])
    h[1] = float(h[1])
    h[2] = float(h[2])
    h[3] = float(h[3])
    data1.append (h)
#Label classes with numbers
for d in range(len(data)):
    if data[d][4] == 'Iris-setosa\n':
        data[d][4] = 0
    elif data[d][4] == 'Iris-versicolor\n':  
        data[d][4] = 1

for d in range(len(data1)):
    if data1[d][4] == 'Iris-setosa\n':
        data1[d][4] = 0
    elif data1[d][4] == 'Iris-versicolor\n':  
        data1[d][4] = 1
        
iris_data = np.array(data)
iris_test = np.array(data1)

#Normalize features with Z-score
for d in range(iris_data.shape[1]-1):
    u = np.mean(iris_data[:,d])
    s = np.std(iris_data[:,d])
    iris_data[:,d] = (iris_data[:,d] - u)/s
    iris_test[:,d] = (iris_test[:,d] - u)/s

#Scatter plots in different feature space
f1 = iris_data[:,0] #Sepal length
f2 = iris_data[:,1] #Sepal width
f3 = iris_data[:,2] #Petal length
f4 = iris_data[:,3] #Petal width
cluster = iris_data[:,4] #Flower class
plt.figure(1)
plt.scatter(f1[cluster==0],f2[cluster==0],marker='+')
plt.scatter(f1[cluster==1],f2[cluster==1],marker='^')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Sepal length vs. Sepal width')
plt.figure(2)
plt.scatter(f1[cluster==0],f3[cluster==0],marker='+')
plt.scatter(f1[cluster==1],f3[cluster==1],marker='^')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Sepal length vs. Petal length')
plt.figure(3)
plt.scatter(f1[cluster==0],f4[cluster==0],marker='+')
plt.scatter(f1[cluster==1],f4[cluster==1],marker='^')
plt.xlabel('Sepal length')
plt.ylabel('Petal width')
plt.title('Sepal length vs. Petal width')
plt.figure(4)
plt.scatter(f2[cluster==0],f3[cluster==0],marker='+')
plt.scatter(f2[cluster==1],f3[cluster==1],marker='^')
plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.title('Sepal width vs. Petal length')
plt.figure(5)
plt.scatter(f2[cluster==0],f4[cluster==0],marker='+')
plt.scatter(f2[cluster==1],f4[cluster==1],marker='^')
plt.xlabel('Sepal width')
plt.ylabel('Petal width')
plt.title('Sepal width vs. Petal width')
plt.figure(6)
plt.scatter(f3[cluster==0],f4[cluster==0],marker='+')
plt.scatter(f3[cluster==1],f4[cluster==1],marker='^')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Petal length vs. Petal width')
#plt.show()

#Append bias to data set
x = -1*np.ones((len(iris_data),1))
a_iris_data = np.concatenate((x, iris_data), 1)
y = -1*np.ones((len(iris_test),1))
a_iris_test = np.concatenate((y, iris_test), 1)
w = [0]*(len(a_iris_data[0])-1)
#Perceptron Gradient Descent 
alpha = 1 #Learning rate
for a in range(30):
    r = randint(0,len(a_iris_data)-1) #randomly choose training examples
    output = a_iris_data[r,0:5].dot(w)
    teacher = a_iris_data[r,5]
    if output >= -w[0]:
        output = 1
    elif output < -w[0]:
        output = 0
    w = w+alpha*(teacher-output)*(a_iris_data[r,0:5]) #delta rule
print(w)
#Testing accuracy
test_output = a_iris_test[:,0:5].dot(w)
for o in range(len(test_output)):
    if test_output[o] >= -w[0]:
        test_output[o] = 1
    elif test_output[o] < -w[0]:
        test_output[o] = 0

err = test_output == a_iris_test[:,5]
err = err.astype(int)
1 - np.mean(err)


# In[ ]:



