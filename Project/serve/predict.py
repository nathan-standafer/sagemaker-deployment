import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

from utils import review_to_words, convert_and_pad

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 500 which represents the converted review
    #         data_len - The length of the review
    
    #print("input_data type: {}".format(type(input_data)))
    #print("input_data: {}".format(input_data))
    
    test_review_list = []
    test_review_words = review_to_words(input_data)
    test_review_list.append(test_review_words)

    #print("type(model.word_dict): {}".format(type(model.word_dict)))
    #print("model.word_dict: {}".format(model.word_dict))
    
    #data_X = convert_and_pad(model.word_dict, test_review_list)[0]  #None  #TypeError: unhashable type: 'list'
    #data_len = convert_and_pad(model.word_dict, test_review_list)[1]
    
    data_X = convert_and_pad(model.word_dict, test_review_words)[0]  #None  #TypeError: unhashable type: 'list'
    data_len = convert_and_pad(model.word_dict, test_review_words)[1]
    
    #print("data_X: {}".format(data_X))
    #print("data_len: {}".format(data_len))
    
    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)
    #print("data_pack: {}".format(data_pack))
    
    data = torch.from_numpy(data_pack)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0
    model_result = model(data) #None
    #print("model_result: {}".format(model_result))
    #print("type(model_result): {}".format(type(model_result)))
    
    result_detached = model_result.cpu().detach()
    #print("result_detached: {}".format(result_detached))
    #print("type(result_detached): {}".format(type(result_detached)))
    
    result = result_detached.numpy()
    #print("result: {}".format(result))
    #print("result.shape: {}".format(result.shape))
    #print("type(result): {}".format(type(result)))
    
    result_rounded = np.rint(result)
    #print("result_rounded: {}".format(result_rounded))
    #print("result_rounded.shape: {}".format(result_rounded.shape))
    #print("type(result_rounded): {}".format(type(result_rounded)))
    
    return_array = np.zeros(1)
    return_array[0] = result_rounded.astype(int)
    #print("return_array: {}".format(return_array))
    #print("return_array.shape: {}".format(return_array.shape))
    #print("type(return_array): {}".format(type(return_array)))
    
    return return_array
