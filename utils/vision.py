import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_attribute_map(image,att):
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(att)
    plt.title('IG Attribute')
    plt.axis('off')

    plt.show()
    
def plot_image_overlay(image,att):
    overlay=np.clip(0.7 * image + 0.5 * att, 0, 1)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    plt.show()

