# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:04:10 2018

@author: Michael
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

#functions for plotting images
def plot_images(imgs, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i], cmap='gray')    

#function for processing an input image
#applies maniupulations
#output a vector
def process_image(img):
    return(img)

def get_whale_categories(labelled_data):
    #for now set it to the id
    labelled_data['category'] = labelled_data['Id']
    return(labelled_data)