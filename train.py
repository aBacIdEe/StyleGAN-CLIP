import torch
import cv2
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn
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


class wordTokenizer(nn.Module):
    pass

    def __init__(self):
        super().__init__()

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        inputs = tokenizer(stuff, return_tensors="pt")
        outputs = model(**inputs)

    def forward(self): # what it does on each call
        # return the token
        last_hidden_states = outputs.last_hidden_state
        pass


class imageTokenizer(nn.Module):
    pass

    def __init__(self):
        super().__init__()

        
    # input image
    # output vector
    pass

def main():
    pass



