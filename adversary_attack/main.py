import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

#modelo preentrenado

from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K

iv3 = InceptionV3()



