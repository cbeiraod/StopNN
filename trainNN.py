'''
Train the Neural Network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.optimizers import Adam, Nadam
import time
import keras
import pandas
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, AlphaDropout
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from commonFunctions import getYields, FullFOM, myClassifier, gridClassifier
#from scipy.stats import ks_2samp
import localConfig as cfg
import pickle
from prepareDATA import *

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Process the command line options')
#   parser.add_argument('-c', '--configFile', required=True, help='Configuration file describing the neural network topology and options as well as the samples to process')
    parser.add_argument('-bt', '--batch', action='store_true', help='Whether this is a batch job, if it is, no interactive questions will be asked and answers will be assumed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-l', '--layers', type=int, required=True, help='Number of layers')
    parser.add_argument('-n', '--neurons', type=int, required=True, help='Number of neurons per layer')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-bs', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-r', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-dc', '--decay', type=float, required=True, help='Learning rate decay')
    parser.add_argument('-dp', '--dropoutRate', type=float, required=True, help='Drop-out rate')

    args = parser.parse_args()

    n_layers = args.layers
    n_neurons = args.neurons
    n_epochs = args.epochs
    batch_size = args.batchSize #len(XDev)/100
    learning_rate = args.learningRate
    my_decay = args.decay
    dropout_rate = args.dropoutRate
    dataset_used = "full+pre" #or "skimmed"

    verbose = 0
    if args.verbose:
        verbose = 1

    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
    myOpt = Adam(lr=learning_rate, decay=my_decay)
    compileArgs['optimizer'] = myOpt

    name = "L"+str(n_layers)+"_N"+str(n_neurons)+"_E"+str(n_epochs)+"_Bs"+str(batch_size)+"_Lr"+str(learning_rate)+"_Dr"+str(dropout_rate)+"_TP"+test_point+"_"+dataset_used

    filepath = cfg.lgbk+"SingleNN/"+name

    if os.path.exists(filepath) == False:
        os.mkdir(filepath)
    os.chdir(filepath)

    if args.verbose:
        print("Dir "+filepath+" created.")
        print("Starting the training")
        start = time.time()
    #call = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=5, verbose=1, mode='auto')
    #model = getDefinedClassifier(len(trainFeatures), 1, compileArgs, n_neurons, n_layers, dropout_rate)
    #model = myClassifier(len(trainFeatures),1, compileArgs, dropout_rate, learning_rate)
    model = gridClassifier(nIn=len(trainFeatures),nOut=1, compileArgs=compileArgs,layers=n_layers,neurons=n_neurons,learn_rate=learning_rate,dropout_rate=dropout_rate)
    history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,shuffle=True, **trainParams)
    acc = history.history["acc"]
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    pickle.dump(acc, open("accuracy.pickle", "wb"))
    pickle.dump(loss, open("loss.pickle", "wb"))
    pickle.dump(val_acc, open("val_accuracy.pickle", "wb"))
    pickle.dump(val_loss, open("val_loss.pickle", "wb"))

    if args.verbose:
        print("Training took ", time.time()-start, " seconds")

    # To save:
    model.save(name+".h5")
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
      json_file.write(model_json)
    model.save_weights(name + ".h5")

    if args.verbose:
        print("Getting predictions")

    devPredict = model.predict(XDev)
    valPredict = model.predict(XVal)

    if args.verbose:
        print("Getting scores")

    scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)

    if args.verbose:
        print "Calculating FOM:"
    dataVal["NN"] = valPredict

    tmpSig, tmpBkg = getYields(dataVal)
    sigYield, sigYieldUnc = tmpSig
    bkgYield, bkgYieldUnc = tmpBkg

    sigDataVal = dataVal[dataVal.category==1]
    bkgDataVal = dataVal[dataVal.category==0]

    fomEvo = []
    fomCut = []

    for cut in np.arange(0.0, 0.9999999, 0.001):
      sig, bkg = getYields(dataVal, cut=cut)
      if sig[0] > 0 and bkg[0] > 0:
        fom, fomUnc = FullFOM(sig, bkg)
        fomEvo.append(fom)
        fomCut.append(cut)

    max_FOM=0

    if args.verbose:
        print "Maximizing FOM"

    for k in fomEvo:
      if k>max_FOM:
        max_FOM=k
    if args.verbose:
        print "Signal@Presel:", sigDataVal.weight.sum() * 35866 * 2
        print "Background@Presel:", bkgDataVal.weight.sum() * 35866 * 2
        print "Signal:", sigYield, "+-", sigYieldUnc
        print "Background:", bkgYield, "+-", bkgYieldUnc

        print "Maximized FOM:", max_FOM
        print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]

    if not args.batch:
        import sys
        import matplotlib.pyplot as plt

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
        plt.savefig(name+'.png')
        #plt.savefig('NN2_'+str(y)+''+str(x)+''+test_point+"_"+str(max_FOM)+'.png

        if args.verbose:
            print "Model name: "+name
        sys.exit("Done!")
