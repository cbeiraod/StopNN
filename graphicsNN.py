#from matplotlib.colors import LogNorm
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import localConfig as cfg
#from prepareDATA import test_point

runNum = 1
test_point = "550_520"
filepath = cfg.lgbk+"Searches/run"+str(runNum)
os.chdir(filepath)

#f = open(filepath+name+'.txt', 'r')
name = "mGS:outputs_run"+str(runNum)+"_"+test_point
f = open(name + '.txt', 'r')

layer = []
neurons = []
roc_AUC = []
ks = []
ks_s = []
ks_b = []
FOM = []
line_index=0

# ===> AQUI CRIA-SE LISTAS DOS VALORES. CONVEM AJUSTAR PAAR QUAIS SE GUARDARAM ETC. NESTE CASO ESTAO 7 A SER GUARDADOS.
for line in f:
    if line_index%7==0:
        layer.append(float(line,))
    if line_index%7==1:
        neurons.append(float(line,))
    if line_index%7==2:
        roc_AUC.append(float(line,))
    if line_index%7==3:
        ks_s.append(float(line,))
    if line_index%7==4:
        ks_b.append(float(line,))
    if line_index%7==5:
        ks.append(float(line,))
    if line_index%7==6:
        FOM.append(float(line,))
    line_index=line_index+1

layers_legend = ["1 layer","2 layers", "3 layers"]
nLayers = len(layers_legend)

plt.figure(figsize=(7,6))
plt.xlabel("Number of Neurons")
plt.ylabel('Roc AUC')
plt.suptitle("Roc curve integral for several configurations of Neural Nets", fontsize=13, fontweight='bold')
plt.title("Learning rate: {0}\nDecay: {1}".format(learning_rate, my_decay), fontsize=10)
# ===> TEM QUE SE AJUSTAR BEM O INICIO E FIM DAS LISTAS PARA CRIAR UMA LINAH POR LAYER/OUTRO PARAMETRO
lim = len(neurons)/nLayers

for i in range(0,nLayers):
    plt.plot(neurons[i*lim:(i+1)*lim], roc_AUC[i*lim:(i+1)*lim])


'''
plt.plot(neurons[0:100], roc_AUC[0:100])
plt.plot(neurons[100:200], roc_AUC[100:200])
plt.plot(neurons[200:300], roc_AUC[200:300])
plt.plot(neurons[300:400], roc_AUC[300:400])
plt.plot(neurons[400:500], roc_AUC[400:500])
plt.plot(neurons[500:600], roc_AUC[500:600])
#plt.axvline(x=25, ymin=0, ymax = 1, linewidth=2, color='black')
#plt.axvline(x=80, ymin=0, ymax = 1, linewidth=2, color='black')
'''
plt.legend(layers_legend, loc='best')
#plt.savefig("roc25_80_"+name+".png")
plt.show()
