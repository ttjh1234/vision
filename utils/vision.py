import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import pylab as P

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

def save_results(path, image,att):
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(att)
    plt.title('IG Attribute')
    plt.axis('off')
    
    
    overlay=np.clip(0.7 * image + 0.5 * att, 0, 1)
    plt.subplot(2,2,3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    igact=np.where(att>=np.quantile(att,0.90),1.0,0.0)
    plt.subplot(2,2,4)
    plt.imshow(igact)
    plt.title('Top 90% Att')
    plt.axis('off')
    
    plt.savefig(path)

def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)


'''
Usage
for i in range(n):
    ROWS = 1
    COLS = 3
    UPSCALE_FACTOR = 20
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    
    tig0=VisualizeImageGrayscale(test_ig[i].transpose(1,2,0))
    tgig0=VisualizeImageGrayscale(test_gig[i].transpose(1,2,0))
    
    ShowImage(rescale_data(test_img[i]), title='Original Image', ax=P.subplot(ROWS, COLS, 1))
    ShowGrayscaleImage(tig0, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))
    ShowGrayscaleImage(tgig0, title='Guided Integrated Gradients', ax=P.subplot(ROWS, COLS, 3))
    
'''