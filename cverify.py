# Used along with the code in https://github.com/ZSusskind/BTHOWeN to verify Cbthowen

from ctypes import *

import sys
import pickle
import lzma
import argparse
import numpy as np
from train_swept_models import get_datasets, binarize_datasets # clone https://github.com/ZSusskind/BTHOWeN

def load_cbthowen(path):
    cbthowen = CDLL(path)

    cbthowen.create_model.argtypes = (c_size_t, c_size_t, c_size_t, c_size_t, c_size_t, c_size_t)
    cbthowen.create_model.restype = None

    cbthowen.set_hashes.argtypes = (POINTER(c_uint64), c_size_t)
    cbthowen.set_hashes.restype = None

    cbthowen.set_ordering.argtypes = (POINTER(c_uint64), c_size_t)
    cbthowen.set_ordering.restype = None

    cbthowen.fill_model.argtypes = (POINTER(c_uint64), c_size_t)
    cbthowen.fill_model.restype = None

    cbthowen.predict.argtypes = (POINTER(c_uint64),)
    cbthowen.predict.restype = c_uint64

    cbthowen.test_write_read.argtypes = None
    cbthowen.test_write_read.restype = None

    cbthowen.test_read_model.argtypes = None
    cbthowen.test_read_model.restype = None

    return cbthowen

def cbthowen_create_model(lib, state_dict):
    lib.create_model(state_dict["num_inputs"], 
        state_dict["bits_per_input"], 
        state_dict["num_classes"], 
        state_dict["num_filter_inputs"], 
        state_dict["num_filter_entries"], 
        state_dict["num_filter_hashes"])

def cbthowen_set_hashes(lib, values):
    inputs = values.flatten().tolist()
    array_type = c_uint64 * len(inputs)

    lib.set_hashes(array_type(*inputs), c_size_t(len(inputs)))

def cbthowen_set_ordering(lib, ordering):
    inputs = ordering.tolist()
    array_type = c_uint64 * len(inputs)

    lib.set_ordering(array_type(*inputs), c_size_t(len(inputs)))

def cbthowen_fill_model(lib, model):
    inputs = []

    for d_it, d in enumerate(model.discriminators):
        for f_it, f in enumerate(d.filters):
            inputs += f.data.tolist()

    array_type = c_uint64 * len(inputs)
    lib.fill_model(array_type(*inputs), c_size_t(len(inputs)))

def cbthowen_predict(lib, input):
    inputs = input.tolist()
    array_type = c_uint64 * len(inputs)

    pred = lib.predict(array_type(*inputs))
    return pred

def cbthowen_write_read(lib):
    lib.test_write_read()

def cbthowen_test_read_model(lib):
    lib.test_read_model()

def verify_inference(lib, inputs, labels, model, bleach=1):
    num_samples = len(inputs)
    # num_samples = 100
    correct = 0
    # ties = 0
    model.set_bleaching(bleach)
    for d in range(num_samples):
        prediction = model.predict(inputs[d])
        cprediction = cbthowen_predict(lib, inputs[d])
        label = labels[d]

        # print(f"label: {label}, cpred: {cprediction}, pred: {prediction}")

        if cprediction == label:
            correct += 1
        # if len(prediction) > 1:
        #     ties += 1
        # if prediction[0] == label:
        #     correct += 1
    correct_percent = round((100 * correct) / num_samples, 4)
    # tie_percent = round((100 * ties) / num_samples, 4)
    print(f"With bleaching={bleach}, accuracy={correct}/{num_samples} ({correct_percent}%)")
    return correct

def verify(model_fname, dset_name, lib_name="cbthowen.so"):
    print("Reading clib")
    cbthowen = load_cbthowen(lib_name)

    print("Loading model")
    with lzma.open(model_fname, "rb") as f:
        state_dict = pickle.load(f)
    if not hasattr(state_dict["model"], "pad_zeros"):
        state_dict["model"].pad_zeros = 0
    
    # cbthowen_create_model(cbthowen, state_dict["info"])
    # cbthowen_set_hashes(cbthowen, state_dict["info"]["hash_values"])
    # cbthowen_set_ordering(cbthowen, state_dict["model"].input_order)
    # cbthowen_fill_model(cbthowen, state_dict["model"])

    # cbthowen_write_read(cbthowen)
    cbthowen_test_read_model(cbthowen)

    print("Loading dataset")
    train_dataset, test_dataset = get_datasets(dset_name)

    bits_per_input = state_dict["info"]["bits_per_input"]
    test_inputs, test_labels = binarize_datasets(train_dataset, test_dataset, bits_per_input)[-2:]

    print("Running inference")
    result = verify_inference(cbthowen, test_inputs, test_labels, state_dict["model"], 1)

def read_arguments():
    parser = argparse.ArgumentParser(description="Verify the implementation of CBTHOWeN")
    parser.add_argument("model_fname", help="Path to pretrained model .pickle.lzma")
    parser.add_argument("dset_name", help="Name of dataset to use for inference; obviously this must match the model")
    parser.add_argument("lib_name", help="Path to the library to verify")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_arguments()
    verify(args.model_fname, args.dset_name, args.lib_name)
