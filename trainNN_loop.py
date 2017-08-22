import root_numpy
import numpy as np
import pandas
import keras
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import localConfig as cfg
from commonFunctions import StopDataLoader
import sys
from math import log

myFeatures = ["LepPt", "LepEta", "LepChg", "Met", "Jet1Pt", "HT", "NbLoose", "Njet", "JetHBpt", "DrJetHBLep", "JetHBCSV", "mt"]
inputBranches = list(myFeatures)
inputBranches.append("XS")
inputBranches.append("weight")
preselection = "(DPhiJet1Jet2 < 2.5 || Jet2Pt < 60) && (Met > 280) && (HT > 200) && (isTight == 1) && (Jet1Pt > 110)"
suffix = "_skimmed"
luminosity = 35866
number_of_events_print = 0
test_point = "550_520"
train_DM = "DM30"

print "Loading datasets..."
dataDev, dataVal = StopDataLoader(cfg.loc, inputBranches, selection=preselection, suffix=suffix, signal=train_DM, test="550_520") #
     
if number_of_events_print == 1:
    np_dataDev, np_dataVal = StopDataLoader(cfg.loc, inputBranches, suffix=suffix, signal=train_DM, test=test_point) #
    print " ==> BEFORE PRE-SELECTION:"        
    print "     Train Signal Events:", len(np_dataDev[np_dataDev.category==1])
    print "     Train Background Events:",len(np_dataDev[np_dataDev.category==0])
    print "     Test Signal Events:", len(np_dataVal[np_dataVal.category==1])
    print "     Test Background Events:", len(np_dataVal[np_dataVal.category==0])        
    print ""
    print " ==> AFTER PRE-SELECTION:"        
    print "     Train Signal Events:", len(dataDev[dataDev.category==1])
    print "     Train Background Events:",len(dataDev[dataDev.category==0])
    print "     Test Signal Events:", len(dataVal[dataVal.category==1])
    print "     Test Background Events:", len(dataVal[dataVal.category==0])

data = dataDev.copy()
data = data.append(dataVal.copy(), ignore_index=True)

print 'Finding features of interest'
trainFeatures = [var for var in data.columns if var in myFeatures]
otherFeatures = [var for var in data.columns if var not in trainFeatures]

print "Preparing the data for the NN"
XDev = dataDev.ix[:,0:len(trainFeatures)] #    PORQUe X E Y??? EQUIVALENTE A SIG E BACK?
XVal = dataVal.ix[:,0:len(trainFeatures)]
YDev = np.ravel(dataDev.category)
YVal = np.ravel(dataVal.category)
weightDev = np.ravel(dataDev.sampleWeight)
weightVal = np.ravel(dataVal.sampleWeight)

print("Fitting the scaler and scaling the input variables")
scaler = StandardScaler().fit(XDev)
XDev = scaler.transform(XDev)
XVal = scaler.transform(XVal)

scalerfile = 'scaler_'+train_DM+'.sav'
joblib.dump(scaler, scalerfile)

compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': 25, 'batch_size': 20, 'verbose': 1}
learning_rate = 0.001/5.0
myAdam = Adam(lr=learning_rate)
compileArgs['optimizer'] = myAdam

## DEFINITIONS
def FOM1(sIn, bIn):
    s, sErr = sIn
    b, bErr = bIn
    fom = s / (b**0.5)
    fomErr = ((sErr / (b**0.5))**2+(bErr*s / (2*(b)**(1.5)) )**2)**0.5
    return (fom, fomErr)

def FOM2(sIn, bIn):
    s, sErr = sIn
    b, bErr = bIn
    fom = s / ((s+b)**0.5)
    fomErr = ((sErr*(2*b + s)/(2*(b + s)**1.5))**2  +  (bErr * s / (2*(b + s)**1.5))**2)**0.5
    return (fom, fomErr)

def FullFOM(sIn, bIn, fValue=0.2):
    s, sErr = sIn
    b, bErr = bIn
    fomErr = 0.0 # Add the computation of the uncertainty later
    fomA = 2*(s+b)*log(((s+b)*(b + (fValue*b)**2))/(b**2 + (s + b) * (fValue*b)**2))
    fomB = log(1 + (s*b*b*fValue*fValue)/(b*(b+(fValue*b)**2)))/(fValue**2)
    fom = (fomA - fomB)**0.5
    return (fom, fomErr)

print "Opening file"
f = open('DATA_loop_test_f_'+test_point+'.txt', 'w')

for y in range(1, 2):
    for x in range(7,8):
        
        if y==1:
            def getDefinedClassifier(nIn, nOut, compileArgs):
                model = Sequential()
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
                model.compile(**compileArgs)
                return model
        if y==2:
            def getDefinedClassifier(nIn, nOut, compileArgs):
                model = Sequential()
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
                model.compile(**compileArgs)
                return model
        if y==3:
            def getDefinedClassifier(nIn, nOut, compileArgs):
                model = Sequential()
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
                model.compile(**compileArgs)
                return model
        if y==4:
            def getDefinedClassifier(nIn, nOut, compileArgs):
                model = Sequential()
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(x, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
                #model.add(Dropout(0.2))
                #model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
                #model.add(Dropout(0.2))
                model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
                model.compile(**compileArgs)
                return model
        
        print("Starting the training")
        call = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=5, verbose=1, mode='auto')
        model = getDefinedClassifier(len(trainFeatures), 1, compileArgs)
        history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,callbacks=[call], **trainParams)

        name = "myNN_"+str(y)+"_"+str(x)+"_"+train_DM
        model.save(name+".h5")
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights(name + ".h5")
        
        # To load:
        #from keras.models import model_from_json
        #with open("myNN_"+str(y)+"_"+str(x)+"_"+test_point+'.json', 'r') as json_file:
        #  loaded_model_json = json_file.read()
        #model = model_from_json(loaded_model_json)
        #model.load_weights("myNN_"+str(y)+"_"+str(x)+"_"+test_point+".h5")
        #model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

        print("Getting predictions")
        devPredict = model.predict(XDev)
        #print devPredict
        valPredict = model.predict(XVal)
        #print valPredict


        print("Getting scores")

        scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 1)
        scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 1)
        print ""
        #print "Dev score:", scoreDev
        #print "Val score:", scoreVal
        #print confusion_matrix(YVal, valPredict.round()) 
        cohen_kappa=cohen_kappa_score(YVal, valPredict.round())


        print "Calculating FOM:"
        dataDev["NN"] = devPredict
        dataVal["NN"] = valPredict
        
        #dataVal[dataVal.category==1].weight = dataVal[dataVal.category==1].weight*dataVal[dataVal.category==1].XS
        sig_dataDev=dataDev[dataDev.category==1]
        bkg_dataDev=dataDev[dataDev.category==0]
        sig_dataVal=dataVal[dataVal.category==1]
        bkg_dataVal=dataVal[dataVal.category==0]

        
        def getYields(dataVal, cut=0.5, luminosity=35866, splitFactor=2):
            #defines the selected test data 
            selectedVal = dataVal[dataVal.NN>cut]
            unselectedVal = dataVal[ dataVal.NN<=cut ]
            
            #separates the true positives from false negatives
            selectedSig = selectedVal[selectedVal.category == 1]
            selectedBkg = selectedVal[selectedVal.category == 0]
            #unselectedSig = unselectedVal[unselectedVal.category == 1]
            #unselectedBkg = unselectedVal[unselectedVal.category == 0]
            
            #print cut
            #print "TP", len(selectedSig)
            #print "FP", len(selectedBkg)
            #print "FN", len(unselectedSig)
            #print "TN", len(unselectedBkg)
            #print ""
            
            sigYield = selectedSig.weight.sum()
            sigYieldUnc = np.sqrt(np.sum(np.square(selectedSig.weight)))
            bkgYield = selectedBkg.weight.sum()
            bkgYieldUnc = np.sqrt(np.sum(np.square(selectedBkg.weight)))

            sigYield = sigYield * luminosity * splitFactor          #The factor 2 comes from the splitting
            sigYieldUnc = sigYieldUnc * luminosity * splitFactor
            bkgYield = bkgYield * luminosity * splitFactor
            bkgYieldUnc = bkgYieldUnc * luminosity * splitFactor

            return ((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))

        tmpSig, tmpBkg = getYields(dataVal)
        sigYield, sigYieldUnc = tmpSig
        bkgYield, bkgYieldUnc = tmpBkg
        

        #print "Signal@Presel:", dataVal[dataVal.category == 1].weight.sum() * 35866 * 2
        #print "Background@Presel:", dataVal[dataVal.category == 0].weight.sum() * 35866 * 2
        #print "Signal:", sigYield, "+-", sigYieldUnc
        #print "Background:", bkgYield, "+-", bkgYieldUnc

        #print "Basic FOM (s/SQRT(b)):", FOM1((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))
        #print "Basic FOM (s/SQRT(s+b)):", FOM2((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))
        #print "Full FOM:", FullFOM((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))

        #sys.exit("Done!")

        #########################################################

        # Let's repeat the above, but monitor the evolution of the loss function


        #history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)
        #print(history.history.keys())

        fomEvo = []
        fomCut = []

        bkgEff = []
        sigEff = []

        sig_Init = dataVal[dataVal.category == 1].weight.sum() * 35866 * 2
        bkg_Init = dataVal[dataVal.category == 0].weight.sum() * 35866 * 2

        for cut in np.arange(0.0, 0.9999999, 0.001):
            #print cut
            sig, bkg = getYields(dataVal, cut=cut, luminosity=luminosity)
            if sig[0] > 0 and bkg[0] > 0:
                fom, fomUnc = FullFOM(sig, bkg)
                fomEvo.append(fom)
                fomCut.append(cut)
                #print fom
                #print ""
                bkgEff.append(bkg[0]/bkg_Init)
                sigEff.append(sig[0]/sig_Init)

        max_FOM=0

        print "Maximizing FOM"
        for k in fomEvo:
            if k>max_FOM:
                max_FOM=k

        Eff = zip(bkgEff, sigEff)

        km_value=ks_2samp((sig_dataDev["NN"].append(bkg_dataDev["NN"])),(sig_dataVal["NN"].append(bkg_dataVal["NN"])))
        
        f.write(str(y)+"\n")           
        print "Layers:", y
        f.write(str(x)+"\n")
        print "Neurons:", x
        f.write(str(cohen_kappa)+"\n")
        print "Cohen Kappa score:", cohen_kappa
        f.write(str(max_FOM)+"\n")
        print "Maximized FOM:", max_FOM
        f.write(str(fomCut[fomEvo.index(max_FOM)])+"\n")
        print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]
        f.write(str(km_value[0])+"\n")
        print "KS test statistic:", km_value[0]
        f.write(str(km_value[1])+"\n")
        print "KS test p-value:", km_value[1]
        
        f.close()
        
        selectedVal = dataVal[dataVal.NN>fomCut[fomEvo.index(max_FOM)]]
        selectedSig = selectedVal[selectedVal.category == 1]
        selectedBkg = selectedVal[selectedVal.category == 0]
        sigYield = selectedSig.weight.sum()
        bkgYield = selectedBkg.weight.sum()
        sigYield = sigYield * luminosity * 2          #The factor 2 comes from the splitting
        bkgYield = bkgYield * luminosity * 2
        
        
        print fomCut[fomEvo.index(max_FOM)]
        print "Number of selected Signal Events:", len(selectedSig)
        print "Number of selected Background Events:", len(selectedBkg)
        print "Sig Yield", sigYield
        print "Bkg Yield", bkgYield
        
        print "Plotting"

        plt.figure(figsize=(7,6))
        plt.hist(sig_dataDev["NN"], 50, facecolor='blue', alpha=0.7, normed=1, weights=sig_dataDev["weight"])
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', alpha=0.7, normed=1, weights=bkg_dataDev["weight"])
        plt.hist(sig_dataVal["NN"], 50, color='blue', alpha=1, normed=1, weights=sig_dataVal["weight"], histtype="step")
        plt.hist(bkg_dataVal["NN"], 50, color='red', alpha=1, normed=1, weights=bkg_dataVal["weight"], histtype="step")
        plt.xlabel('NN output')
        #plt.title("Cohen's kappa: {0}".format(cohen_kappa), fontsize=10)
        plt.suptitle("MVA overtraining check for classifier: NN", fontsize=13, fontweight='bold')
        plt.title("Cohen's kappa: {0}\nKolmogorov Smirnov test: {1}".format(cohen_kappa, km_value[1]), fontsize=10)
        plt.legend(['Signal (Test sample)', 'Background (Test sample)', 'Signal (Train sample)', 'Background (Train sample)\nasdfgh'], loc='upper right')
        plt.savefig('hist_'+str(y)+'_'+str(x)+'_'+test_point+'.png', bbox_inches='tight')
        plt.show()


        both_dataDev=bkg_dataDev.append(sig_dataDev)
        plt.figure(figsize=(7,6))
        plt.xlabel('NN output')
        plt.title("Number of Events")
        #plt.yscale('log', nonposy='clip')
        plt.legend(['Background + Signal (test sample)', 'Background (test sample)'], loc="upper left" )
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', weights=bkg_dataDev["weight"])
        plt.hist(both_dataDev["NN"], 50, color="blue", histtype="step", weights=both_dataDev["weight"])
        plt.savefig('pred_'+str(y)+'_'+str(x)+'_'+test_point+'.png', bbox_inches='tight')
        plt.show()


        plt.figure(figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(211)
        plt.plot(fomCut, fomEvo)
        plt.title("FOM")
        plt.ylabel("FOM")
        plt.xlabel("ND")
        plt.legend(["Max. FOM: {0}".format(max_FOM)], loc='upper left')


        plt.subplot(212)
        plt.semilogy(fomCut, Eff)
        plt.axvspan(fomCut[fomEvo.index(max_FOM)], 1, facecolor='#2ca02c', alpha=0.3)
        #plt.axvline(x=fomCut[fomEvo.index(max_FOM)], ymin=0, ymax=1)
        plt.title("Efficiency")
        plt.ylabel("Eff")
        plt.xlabel("ND")
        plt.legend(['Background', 'Signal'], loc='upper left')
        plt.savefig('FOM_'+str(y)+'_'+str(x)+'_'+test_point+'.png', bbox_inches='tight')
        plt.show()

    


sys.exit("Done!")

