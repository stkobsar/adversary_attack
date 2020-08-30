from keras.applications.inception_v3 import InceptionV3, decode_predictions


def get_prediction(image_processed):
    iv3 = InceptionV3()
    image_pred = iv3.predict(image_processed)
    image_pred = decode_predictions(image_pred)
    return image_pred