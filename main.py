import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image as kp_image
from tkinter.filedialog import askopenfilename, asksaveasfilename
from window import Window
from image_proc import *
from model import *

if __name__ == '__main__':
    print('Start')

window = Window()
window.show()


