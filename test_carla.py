#! /usr/bin/env python

"""
Lane Detection

    Usage: python3 test.py  --conf=./config.json

"""


import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
import argparse
import json
from src.frontend import Segment
from quant_mode import quant_model

# define command line arguments
argparser = argparse.ArgumentParser(
    description='Evaluate Road Segmentation Model')

argparser.add_argument(
    '-c',
    '--conf', default="config_UNet.json",
    help='path to configuration file')


def lane_detect(image):
    """
    :param args: command line argument
    """

    # Parse command line argument
    args = argparser.parse_args()
    config_path = args.conf

    # Open and load the config json
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # parse the json to retrieve the training configuration
    backend = config["model"]["backend"]
    input_size = (config["model"]["im_width"], config["model"]["im_height"])
    classes = config["model"]["classes"]

    # define the model and train
    segment = Segment(backend, input_size, classes)
    model = segment.feature_extractor
    # Load best model
    model.load_weights(config['test']['model_file'])
    # model = quant_model(model)
    # print(model.summary())
    pred = model.predict(image)
    return pred