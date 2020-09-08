import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

### Description

## I train my keras model with Tensorflow 2.1 backend, in a non-CMSSW area, with python
## I have saved my .h5 and I want to convert it to a graph

# for tf v2, fgo into v1 compatibility mode
if tf.__version__.startswith("2."):
    tf = tf.compat.v1
tf.disable_eager_execution()

sess = tf.Session()

model = keras.models.load_model('../nn_inference/model_2.h5')

import tensorflow.python.keras.backend as K
K.set_session(sess)

#session = keras.backend.get_session()
## # Note:
## # The previous line causes errors:
## # Error while reading resource variable dense_3/bias from Container: localhost.
## # This could mean that the variable was uninitialized. 
## # Not found: Container localhost does not exist.

## # This line works
init = tf.global_variables_initializer()
sess.run(init)

model.summary()
print("Model outputs")
print([node.op.name for node in model.outputs])
print("Model inputs")
print([node.op.name for node in model.inputs])

outputs = [node.op.name for node in model.outputs]
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)

## This works but when used in C++ it gives completely different predictions
#tf.train.write_graph(constant_graph, "../nn_inference/", "constantgraph.pb", as_text=False)

## This also works but gives completely different predictions w.r.t. previous line, and also w.r.t python
graph_io.write_graph(constant_graph, "../nn_inference/", "constantgraph.pb", as_text=False)
