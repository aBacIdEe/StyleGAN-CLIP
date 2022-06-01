import torch
import cv2
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
class config:
    image_path = "dataset"
    # batch size
    # epochs
    # other parameters

# Classes we'll probably need

class Dataset: # maybe inherit the pytorch Dataset class
    '''
    Properties of a Dataset:
    A list to hold the image-text pairs
    '''
    pass

class CLIPModel:
    '''
    Properties of the model
    its nn
    batch size
    epochs
    checkpoints maybe
    '''

    '''
    methods of the model
    Calculate the Cost
    Update the parameters
    '''
    pass

# Functions we'll probably need but just not in a class

def imageTokenizer():
    # input image 
    # output vector 
    pass

def textTokenizer():
    # input string
    # output vector
    pass

def main():
    pass



