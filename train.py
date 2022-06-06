import torch
import cv2
import numpy as np
import torchvision.models as models
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

class CLIPModel(nn.Module):
    
    def __init__(self):
        super().__init__()


    '''
    methods of the model
    Foward: 
    Calculate the Cost
    Update the parameters
    '''
    pass

'''
Somehow setup these tokenizers
'''
class wordTokenizer(nn.Module):
    pass

    def __init__(self, input):
        super().__init__()

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.inputs = tokenizer(input, return_tensors="pt")
        

    def forward(self): # what it does on each call
        # return the token
        outputs = self.model(**self.inputs)
        last_hidden_states = outputs.last_hidden_state
        pass


class imageTokenizer(nn.Module):
    pass

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        
    # input image
    # output vector
    def forward(self, input):
        return self.model(input)

def train():
    pass

def main():
    pass



