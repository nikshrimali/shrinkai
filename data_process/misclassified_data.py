# Code that gets the images that are misclassified by our model and convert it into a single figure

import numpy as np
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()

def plot_misclassified(misclassified:list):    
    plt.figure(figsize=(20,20))
    for plotIndex, badIndex in enumerate(misclassified[0:25]):
        plt.subplot(5, 5, plotIndex + 1,)
        plt.axis('off')
        plt.imshow(np.reshape(misclassified[plotIndex][0].cpu(), (28,28)), cmap=plt.cm.gray)
        plt.title("Predicted: {}, Actual: {}".format(misclassified[plotIndex][1], misclassified[plotIndex][2].cpu().numpy(), fontsize = 20))
        plt.savefig(os.path.join("..",'misclassified.png'))
