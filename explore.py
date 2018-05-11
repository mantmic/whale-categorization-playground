# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:00:35 2018

@author: Michael
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

import whale_functions

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

np.random.seed(20)

training_data = pd.read_csv('data/train.csv')

#check how many types of whale there are and how frequently they occur
whale_count =  training_data.Id.value_counts()

#filter out the new whales, the most common group
whale_count = whale_count[1:]

#how many just have one occurance?
sum(whale_count == 1)

#about half just have one occurance. 


#Data augmentation 

#seeing as there can be 5 predictions per whale, perhaps these should be clustered into similar whales
#then treated as a single whale type

sum(whale_count >= 4)

#only 504 whales have 4 or more images.
#let's cluster any whale that has less than 3 images into whale categories containing 5 whales
#other ones can be treated as their own whale categories

#commonly found whales will have 1 prediction, rare ones will have 5, boosting the precision score
#to increase score the number of whales in a cluster should later be decreased

#create a dataset with whale id, whale categories

#labels will be whale categories

#That will be round 2, after we have a base model


#as for image processing, noise from the ocean should be elimiated from the image, this places a heavier emphasis on the whale.

#perhaps portion of image that is ocean should be identified using images of oceans, then removed from the picture?
#object detection of ocean may be advantageous 

#this would take while, let's consider it a round 3

