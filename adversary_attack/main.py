import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import adversary_attack.functions.get_image as gt
import adversary_attack.functions.get_prediction as gp
import adversary_attack.functions.adversary_image as aa
import adversary_attack.functions.get_class_adv_pred as gc
from PIL import Image
from keras_preprocessing import image

def plot_image_adv(image_denormalized):
    plt.imshow(image_denormalized[0].astype(np.uint8))
    image = plt.savefig("test_1")
    return image

def parse_argse():
    parser = argparse.ArgumentParser(description='get image to procces it in iv3')
    parser.add_argument('-ip', '--image_path', type=str, help='image path to process', default="data")
    args = parser.parse_args()
    return args.image_path


def main(image_path):
    absolute_path_every_machine = os.path.abspath(__file__)
    adv_dir = os.path.dirname(os.path.dirname(absolute_path_every_machine))
    image_relative_path = image_path
    image_path = os.path.join(adv_dir, image_relative_path)
    image_processed = gt.get_image(image_path)

    image_pred_matrix = gp.get_prediction_numeric(image_processed)


    return image_processed


if __name__ == "__main__":

    image_path = parse_argse()
    image_processed = main(image_path)

    image_pred_copy = aa.adversary(image_processed)

    image_denormalized = gt.denormalize_matrix_image(image_pred_copy)
    adversary_image = plot_image_adv(image_denormalized)




    ########## Show image with moved pixels ###############







