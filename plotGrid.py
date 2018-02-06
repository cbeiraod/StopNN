'''
Plot the results of a Grid Search
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import localConfig as cfg

def findParam(inA,word,outA):
    for i in range(4,len(inA)):
        if inA[i].find(word) != -1:
            outA.append(float(inA[i+1].replace(',','').replace('}','')))

filepath = cfg.lgbk+"Searches"
os.chdir(filepath)

with open("gS:2017-09-06_00:10.txt") as f:
    data = f.read()

data = data.split("\n")

accuracy = []
std_accuracy = []
layers = []
learn_rate = []
dropout_rate = []
batch_size = []
epochs = []
neurons = []

for line in range(5,len(data)-1):
    l = data[line].split(" ")
    accuracy.append(float(l[0]))
    std_accuracy.append(float(l[2].replace('(','').replace(')','')))

    findParam(l,'layers',layers)
    findParam(l,'learn_rate',learn_rate)
    findParam(l,'dropout_rate',dropout_rate)
    findParam(l,'batch_size',batch_size)
    findParam(l,'epochs',epochs)
    findParam(l,'neurons',neurons)

di = dropout_rate.index(0.5)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(121, projection='3d')
x = neurons[:di]
y = layers[:di]
z = accuracy[:di]
ax.plot(x, y, z, c='r', marker='o')
ax.set_title('Dropout-rate: 0')
ax.set_xlabel('Neurons')
ax.set_ylabel('Layers')
ax.set_zlabel('Accuracy')

ax = fig.add_subplot(122, projection='3d')
x = neurons[di:]
y = layers[di:]
z = accuracy[di:]
ax.scatter(x, y, z, c='b', marker='^')
ax.set_title('Dropout-rate: 0.5')
ax.set_xlabel('Neurons')
ax.set_ylabel('Layers')
ax.set_zlabel('Accuracy')

plt.show()

'''
print accuracy
print "\n"
print std_accuracy
print "\n"
print layers
print "\n"
print learn_rate
print "\n"
print dropout_rate
print "\n"
print batch_size
print "\n"
print epochs
print "\n"
print neurons
'''
