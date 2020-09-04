#!/usr/bin/env python
# coding: utf-8

from pandas import read_csv, DataFrame
from keras.models import load_model
from flask import Flask,request,jsonify, make_response
import tensorflow as tf
import json
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
import pickle
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import set_session

from util import *

application = Flask(__name__)  

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

sess = tf.Session()
graph = tf.compat.v1.get_default_graph()

set_session(sess)
model = tf.keras.models.load_model('ML_Model/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})

xgb_model = pickle.load(open('ML_Model/xgb_clf.pickle.dat', 'rb'))

embedding_dim = 50
max_seq_length = 20
xgb_weight = 0.7

@application.route('/get_sentence_similarity', methods = ['POST'])
def sentenceSimilarity():
    global model, graph, xgb_model, sess
    
    if 'injson' not in request.form:
        return make_response(jsonify([{"error": "Request body absent"}]), 400)
    
    try:
        json_data = json.loads(request.form['injson'])
    except Exception as e:
        return make_response(jsonify([{"error": "Error parsing JSON"}]), 400)
    
    if 'Sentence1' not in json_data or 'Sentence2' not in json_data:
        return make_response(jsonify([{"error": "Invalid JSON"}]), 400)
    
    data = DataFrame(json_data, index=[0])
    data_lstm = data.copy()
    
    data['word_shares'] = data.apply(word_shares, axis=1)
    
    data = get_xgb_features(data)
    
    try:
        xgb_pred = xgb_model.predict_proba(data)[:,1]
    except Exception as e:
        return make_response(jsonify([{"error": "Model prediction failed"}]), 500)
    
    for s in ['Sentence1', 'Sentence2']:
        data_lstm[s + '_n'] = data_lstm[s]
        
    data_lstm, embeddings_test = make_w2v_embeddings(data_lstm, embedding_dim=embedding_dim)
    
    data_lstm = split_and_zero_padding(data_lstm, max_seq_length)
    
    with graph.as_default():
        set_session(sess)
        try:
            lstm_pred = model.predict([data_lstm['left'], data_lstm['right']])
        except Exception as e:
            return make_response(jsonify([{"error": "Model prediction failed"}]), 500)    
    
    probability = lstm_pred[...,0]*(1-xgb_weight) + xgb_pred[0]*xgb_weight
    
    y_pred = np.where(probability > 0.5, 1, 0)
    
    json_data['isSimilar'] = int(y_pred[0])
    
    return make_response(jsonify([json_data]), 200)


if __name__ == '__main__':
    application.run(host= '0.0.0.0',port=4002)

