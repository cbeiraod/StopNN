"""
Make a Grid Search varying the number of neurons per layer and the number of layers.
Plot the results to using the area under the ROC curve as a metric of performance.
"""


import root_numpy
import os
import numpy as np
import pandas
import keras
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.optimizers import Adam, Nadam
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import ks_2samp
#import matplotlib.pyplot as plt
import localConfig as cfg
from commonFunctions import StopDataLoader, FullFOM, getYields
import pickle
import sys
from math import log

from prepareDATA import *

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Process the command line options')
#   parser.add_argument('-c', '--configFile', required=True, help='Configuration file describing the neural network topology and options as well as the samples to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-r', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-d', '--decay', type=float, required=True, help='Learning rate decay')
    parser.add_argument('-l', '--layers', type=int, required=False, help='Number of layers')
    parser.add_argument('-n', '--neurons', type=int, required=False, help='Number of neurons per layer')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-bs', '--batchSize', type=int, required=True, help='Batch size')

    args = parser.parse_args()

    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    trainParams = {'epochs': args.epochs, 'batch_size': args.batchSize, 'verbose': 0}
    learning_rate = args.learningRate
    my_decay = args.decay
    myAdam = Adam(lr=learning_rate, decay=my_decay)
    compileArgs['optimizer'] = myAdam

    print "Opening file"

    runNum = 1
    filepath = cfg.lgbk+"Searches/"

    while os.path.exists(filepath+"run"+str(runNum)):
        runNum += 1

    filepath = filepath+"run"+str(runNum)

    os.mkdir(filepath)
    os.mkdir(filepath+"/accuracy")
    os.mkdir(filepath+"/loss")
    os.chdir(filepath)

    name = "mGS:outputs_run"+str(runNum)+"_"+test_point+"_"+str(learning_rate)+"_"+str(my_decay)
    f = open(name+'.txt', 'w')

    for y in [1,2,3]:   # LAYERS
        for x in range(2, 101):    # NEURONS
            print "  ==> #LAYERS:", y, "   #NEURONS:", x, " <=="

            print("Starting the training")
            model = getDefinedClassifier(len(trainFeatures), 1, compileArgs, x, y)
            history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)

    	name = "L"+str(y)+"_N"+str(x)+"_"+train_DM+"_run"+str(runNum)

    	acc = history.history["acc"]
        #val_acc = history.history['val_acc']
        loss = history.history['loss']
        #val_loss = history.history['val_loss']
        pickle.dump(acc, open("accuracy/acc_"+name+".pickle", "wb"))
        pickle.dump(loss, open("loss/loss_"+name+".pickle", "wb"))
        model.save(name+".h5")
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights(name + ".h5")

        print("Getting predictions")
        devPredict = model.predict(XDev)
        valPredict = model.predict(XVal)

        print("Getting scores")

        scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
        scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)
        cohen_kappa=cohen_kappa_score(YVal, valPredict.round())

        print "Calculating FOM:"
        dataDev["NN"] = devPredict
        dataVal["NN"] = valPredict

        sig_dataDev=dataDev[dataDev.category==1]
        bkg_dataDev=dataDev[dataDev.category==0]
        sig_dataVal=dataVal[dataVal.category==1]
        bkg_dataVal=dataVal[dataVal.category==0]


        tmpSig, tmpBkg = getYields(dataVal)
        sigYield, sigYieldUnc = tmpSig
        bkgYield, bkgYieldUnc = tmpBkg

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

        max_FOM=0

        print "Maximizing FOM"
        for cv_0 in fomEvo:
            if cv_0>max_FOM:
                max_FOM=cv_0

        roc_Integral = 0
        for cv_1 in range(0, len(bkgEff)-1):
            roc_Integral=roc_Integral+0.5*(bkgEff[cv_1]-bkgEff[cv_1+1])*(sigEff[cv_1+1]+sigEff[cv_1])

        Eff = zip(bkgEff, sigEff)

        km_value_s = ks_2samp(sig_dataDev["NN"], sig_dataVal["NN"])[1]
        km_value_b = ks_2samp(bkg_dataDev["NN"], bkg_dataVal["NN"])[1]
        km_value = ks_2samp(dataDev["NN"], dataVal["NN"])[1]

        f.write(str(y)+"\n")
        f.write(str(x)+"\n")
        f.write(str(roc_Integral)+"\n")
        f.write(str(km_value_s)+"\n")
        f.write(str(km_value_b)+"\n")
        f.write(str(km_value)+"\n")
        f.write(str(max_FOM)+"\n")

    sys.exit("Done!")

def getDefinedClassifier(nIn, nOut, compileArgs, neurons=12, layers=1):
model = Sequential()
model.add(Dense(neurons, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
    for x in range(0, layers-1):
        model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
model.compile(**compileArgs)
return model
