#from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

f = open('DATA_loop_test_trash_550_520.txt', 'r')

layers = []
neurons = []
cohen_kappa =[]
FOM_max = []
FOM_cut = []
KS_test_stat = []
KS_test_pval = []
layers_legend = []
line_index=0

for line in f:
    if line_index%7==0:
        layers.append(float(line,))
    if line_index%7==1:
        neurons.append(float(line,))
    if line_index%7==2:
        cohen_kappa.append(float(line,))
    if line_index%7==3:
        FOM_max.append(float(line,))
    if line_index%7==4:
        FOM_cut.append(float(line,))
    if line_index%7==5:
        KS_test_stat.append(float(line,))
    if line_index%7==6:
        KS_test_pval.append(float(line,))
    line_index=line_index+1

        
for l in list(set(layers)):   
    layers_legend.append(str(l)+" layers")
            
           
nol = len(list(set(layers)))
non = len(list(set(neurons)))



plt.figure(figsize=(7,6))
plt.xlabel('Number of neurons per layer')
plt.ylabel('F.O.M.')
#plt.title("Cohen's kappa: {0}".format(cohen_kappa), fontsize=10)
plt.suptitle("FOM for several configurations of Neural Nets", fontsize=13, fontweight='bold')
#plt.title("Cohen's kappa: {0}\nKolmogorov Smirnov test: {1}".format(cohen_kappa, km_value[1]), fontsize=10)
for x in range(0, nol):
    plt.plot(neurons[int(x*non):int((x+1)*non)], FOM_max[int(x*non):int((x+1)*non)])
#plt.plot(neurons[0:18], FOM_max[0:18], "b")
#plt.plot(neurons[18:36], FOM_max[18:36], "r")
#plt.plot(neurons[36:54], FOM_max[36:54], "g")
#plt.plot(neurons[54:72], FOM_max[54:72], "c")
plt.legend(layers_legend, loc='best')
plt.show()

plt.hist2d(neurons, layers, bins=[non,nol], weights=FOM_max)
plt.colorbar()
plt.show()