from PIL import Image
import PIL
import adversary_attack.functions.get_prediction as gp

def adv_test_class(image):
    adv_image = gp.get_prediction_class(image)
    return adv_image


"""
if __name__ == "__main__":

    # creating a image object (main image)
    filename = "Users/Stephi/Documents/Programming/REPOSITORIES/adversary_attack/data/test_1.png"

    im1 = Image.open(filename)
    print(im1)
    # save a image using extension
    im1 = im1.save("geeks.jpg")
    open(im1)
"""

