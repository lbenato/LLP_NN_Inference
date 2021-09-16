import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
import cmsml

#from tensorflow.python.framework import graph_util
#from tensorflow.python.framework import graph_io

# for tf v2, fgo into v1 compatibility mode
#if tf.__version__.startswith("2."):
#    tf = tf.compat.v1
#tf.disable_eager_execution()

#sess = tf.Session()

#model = keras.models.load_model('model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel_50_50/model_FCN_2_EventWeightNormalized_NoMedian_Adam_patience40_batch_size_2048/model_2_EventWeightNormalized_NoMedian_Adam_patience40_batch_size_2048.h5')

#example
#model = keras.models.load_model('nn_inference/model_2.h5')
#updated model
#model = keras.models.load_model('nn_inference/model_2_updated.h5')
#model = keras.models.load_model('nn_inference/tagger_AK8_v1/model.h5')
#model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')
#model = keras.models.load_model('nn_inference/tagger_AK4_miniAOD_v3/model.h5')
model = keras.models.load_model('nn_inference/tagger_AK8_v2_double_match/model.h5')


#import tensorflow.python.keras.backend as K
#K.set_session(sess)

#session = keras.backend.get_session()
## # Causing error:
## # Error while reading resource variable dense_3/bias from Container: localhost.
## # This could mean that the variable was uninitialized. 
## # Not found: Container localhost does not exist.
#init = tf.global_variables_initializer()
#sess.run(init)

print('\n')
model.summary()
#print("All nodes in graph")
#print([n.name for n in sess.graph.as_graph_def().node])
print("Model outputs")
print([node.op.name for node in model.outputs])
print("Model inputs")
print([node.op.name for node in model.inputs])
#print("Joerg")
#for i in tf.get_default_graph().get_operations():
#    print(i.name)

print('\n')
cmsml.tensorflow.save_graph("nn_inference/tagger_AK8_v2_double_match/graph.pb", model, variables_to_constants=True)
cmsml.tensorflow.save_graph("nn_inference/tagger_AK8_v2_double_match/graph.pb.txt", model, variables_to_constants=True)

#outputs = [node.op.name for node in model.outputs]
#min_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [node.op.name for node in model.outputs])

#outputs = ['dense_4_target']#['dense_4/Softmax']#
#constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
#was working
#tf.train.write_graph(constant_graph, "", "constantgraph.pb", as_text=False)
#graph_io.write_graph(constant_graph, "", "constantgraph.pb", as_text=False)
