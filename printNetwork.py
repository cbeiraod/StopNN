import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import localConfig as cfg
from keras.models import model_from_json
from keras.utils import plot_model

#runNum = 20
#model_name = "L1_N5_550_520_run20"
#filepath = cfg.lgbk + "Searches/run"+str(runNum)
model_name = "newDATA_N25_L2_E300_B30000_Lr0.003_Dr0.0_Dev550_520_Val550_520"
filepath = "/home/diogo/LIP/LogBook/SingleNN/"+model_name
os.chdir(filepath)

print "Loading Model ..."
with open(model_name+'.json', 'r') as json_file:
  loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+".h5")

#apt-get install -y graphviz libgraphviz-dev

plot_model(model, to_file='model.png', show_shapes=True)
