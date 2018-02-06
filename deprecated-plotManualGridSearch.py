import os
import keras
import matplotlib.pyplot as plt
import localConfig as cfg
from keras.models import model_from_json, Sequential
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, roc_auc_score
from prepareDATA import *
from commonFunctions import FOM1, FOM2, FullFOM, getYields
#StopDataLoader

runNum = 3
model_name = "L2_N25_550_520_run3"

filepath = cfg.lgbk+"Searches/run"+str(runNum)
os.chdir(filepath)

print "Loading Model ..."
with open(model_name+'.json', 'r') as json_file:
  loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+".h5")
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

print("Getting predictions")
devPredict = model.predict(XDev)
valPredict = model.predict(XVal)

print "Calculating parameters"
dataDev["NN"] = devPredict
dataVal["NN"] = valPredict

sig_dataDev=dataDev[dataDev.category==1]
bkg_dataDev=dataDev[dataDev.category==0]
sig_dataVal=dataVal[dataVal.category==1]
bkg_dataVal=dataVal[dataVal.category==0]

fomEvo = []
fomCut = []

bkgEff = []
sigEff = []

sig_Init = dataVal[dataVal.category == 1].weight.sum() * 35866 * 2
bkg_Init = dataVal[dataVal.category == 0].weight.sum() * 35866 * 2

for cut in np.arange(0.0, 0.9999, 0.001):
    sig, bkg = getYields(dataVal, cut=cut, luminosity=luminosity)
    if sig[0] > 0 and bkg[0] > 0:
        fom, fomUnc = FullFOM(sig, bkg)
        fomEvo.append(fom)
        fomCut.append(cut)
        bkgEff.append(bkg[0]/bkg_Init)
        sigEff.append(sig[0]/sig_Init)


print "Plotting"
plt.figure(figsize=(7,6))
plt.hist(sig_dataDev["NN"], 50, facecolor='blue', alpha=0.7, normed=1, weights=sig_dataDev["weight"])
plt.hist(bkg_dataDev["NN"], 50, facecolor='red', alpha=0.7, normed=1, weights=bkg_dataDev["weight"])
plt.hist(sig_dataVal["NN"], 50, color='blue', alpha=1, normed=1, weights=sig_dataVal["weight"], histtype="step")
plt.hist(bkg_dataVal["NN"], 50, color='red', alpha=1, normed=1, weights=bkg_dataVal["weight"], histtype="step")
plt.xlabel('NN output')
plt.suptitle("MVA overtraining check for classifier: NN", fontsize=13, fontweight='bold')
#plt.title("Roc Curve AUC: {0} \nKolmogorov Smirnov test (s,b,s+b): ({1}, {2}, {3})".format(roc_Integral, km_value_s, km_value_b, km_value), fontsize=10)
plt.legend(['Signal (Test sample)', 'Background (Test sample)', 'Signal (Train sample)', 'Background (Train sample)'], loc='upper right')
plt.savefig('hist_'+model_name+'.png', bbox_inches='tight')

plt.figure(figsize=(7,6))
plt.plot(bkgEff, sigEff)
plt.title("Roc curve", fontweight='bold')
plt.ylabel("Signal efficiency")
plt.xlabel("Bakcground efficiency")
plt.axis([0, 1, 0, 1])
#plt.legend(["Roc curve integral: {0}".format(roc_Integral)], loc='best')
plt.savefig('Roc_'+model_name+'.png', bbox_inches='tight')

print 

'''
plt.figure(figsize=(7,6))
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("evo_"+name+'.png')
#plt.show()
