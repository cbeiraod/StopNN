import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import localConfig as cfg
from keras.models import model_from_json
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap
import localConfig as cfg

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Process the command line options')
#   parser.add_argument('-d', '--dryRun', action='store_true', help='Do a dry run (i.e. do not actually run the potentially dangerous commands but print them to the screen)')
#   parser.add_argument('-c', '--configFile', required=True, help='Configuration file describing the neural network topology and options as well as the samples to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
#   parser.add_argument('-o', '--outDirectory', required=True, help='Name of the output directory')
    parser.add_argument('-b', '--batch', action='store_true', help='Whether this is a batch job, if it is, no interactive questions will be asked and answers will be assumed')
    parser.add_argument('-f', '--file', help='File name')
    parser.add_argument('-l', '--layers', type=int, required=False, help='Number of layers')
    parser.add_argument('-n', '--neurons', type=int, required=False, help='Number of neurons per layer')
    parser.add_argument('-ll', '--local', action='store_true', help='Run locally')
    parser.add_argument('-s', '--singleNN', action='store_true', help='Whether this NN is stored in the Searches or SingleNN folder')
    parser.add_argument('-r', '--runNum', type=int, help='Run number')

    args = parser.parse_args()

    if args.file != None:
        model_name = args.file
    else:
        model_name = "L"+str(args.layers)+"_N"+str(args.neurons)+"_550_520_run"+str(args.runNum)

    if args.singleNN:
        filepath = cfg.lgbk + "SingleNN/" + model_name
    elif args.runNum != None:
        filepath = cfg.lgbk + "Searches/run" + str(args.runNum)
    elif args.local:
        filepath = "/home/diogo/PhD/SingleNN/" + model_name

    os.chdir(filepath)

    with open(model_name+'.json', 'r') as json_file:
      loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name+".h5")

    cdict = {'red':   ((0.0, 0.97, 0.97),
                       (0.25, 0.0, 0.0),
                       (0.75, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.25, 0.25),
                       (0.25, 0.15, 0.15),
                       (0.75, 0.39, 0.39),
                       (1.0, 0.78, 0.78)),

             'blue':  ((0.0, 1.0, 1.0),
                       (0.25, 0.65, 0.65),
                       (0.75, 0.02, 0.02),
                       (1.0, 0.0, 0.0))
            }
    myColor = LinearSegmentedColormap('myColorMap', cdict)

    nLayers = 0
    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
        nLayers+=1

    maxWeights = 0

    '''
    plt.imshow(model.layers[0].get_weights()[0])
    plt.show()
    '''
    figure = plt.figure()
    figure.suptitle("Weights", fontsize=12)

    i=1
    nRow=2
    nCol=3

    if nLayers < 5:
        nRow = 2.0
        nCol = 2

    elif nLayers < 10:
        nRow = math.ceil(nLayers / 3)
        nCol = 3

    else:
        nRow = math.ceil(nLayers / 4)
        nCol = 4

    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
        ax = figure.add_subplot(nRow, nCol,i)

        im = plt.imshow(layer.get_weights()[0], interpolation="none", vmin=-2, vmax=2, cmap=myColor)
        plt.title(layer.name, fontsize=10)
        plt.xlabel("Neuron", fontsize=9)
        plt.ylabel("Input", fontsize=9)
        plt.colorbar(im, use_gridspec=True)

        i+=1

    plt.tight_layout()
    plt.savefig('Weights_'+model_name+'.pdf', bbox_inches='tight')
    plt.show()
