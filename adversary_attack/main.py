import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import adversary_attack.functions.get_image as gt
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras_preprocessing import image
from keras import backend as K


def parse_argse():
    parser = argparse.ArgumentParser(description='get image to procces it in iv3')
    parser.add_argument('--image_path', type=str, help='image path to process', default="data")
    args = parser.parse_args()
    return args.image_path


def main(image_path):
    absolute_path_every_machine = os.path.abspath(__file__)
    adv_dir = os.path.dirname(os.path.dirname(absolute_path_every_machine))
    image_relative_path = image_path
    image_path = os.path.join(adv_dir, image_relative_path)
    image_processed = gt.get_image(image_path)
    return image_processed


if __name__ == "__main__":
    image_path = parse_argse()
    image_processed = main(image_path)
    print(image_processed)



