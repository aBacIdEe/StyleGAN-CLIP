from cgitb import text
import cv2
import numpy as np
import pandas as pd
import os
from tqdm.autonotebook import tqdm
import albumentations as A # provides fast image augmentation and implements image transform
import torchvision.models as models
from transformers import DistilBertTokenizer, DistilBertModel

import torch
from torch import nn
import torch.nn.functional as F

class Config:
    debug = False
    image_path = "Datasets/Flicker-30k/Images"
    captions_path = "Datasets/Flicker-30k"
    max_length = 32
    size = 0 #TODO
    projection_dim = 256
    temperature = 1
    image_embedding = 2048
    text_embedding = 768
    batch_size = 32
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.1

# Classes we'll probably need

class Dataset(torch.utils.data.Dataset):

    def __init__(self, files, captions, tokenizer, transforms):
        self.files = files
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=Config.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{Config.image_path}/{self.files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)

def get_transformers(mode="train"):
    return A.Compose(
        [
            A.Resize(Config.size, Config.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True)
        ]
    ) 


    '''
    Properties of a Dataset:
    A list to hold the image-text pairs
    '''

class CLIPModel(nn.Module):
    
    def __init__(
        self, temperature=Config.temperature, 
        image_embedding=Config.image_embedding, 
        text_embedding=Config.text_embedding):

        super().__init__()
        
        # encoders
        self.image_encoder = ImageTokenizer()
        self.text_encoder = WordTokenizer()
        
        # to project both image and text onto the same 256 plane
        self.image_projection = Projection(embedding_dim=image_embedding)
        self.text_projection = Projection(embedding_dim=text_embedding)
        
        self.temperature = temperature
    
    def forward(self, input):
        image_features = self.image_encoder(input["image"])
        # getting text labels
        text_features = self.text_encoder(
            input_ids = input["input_ids"], attention_mask=input["attention_mask"]
        )

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        logit = image_embeddings @ text_embeddings.T
        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax(
            (image_similarity + text_similarity) / 2 , dim=-1)
        image_loss = self.cross_entropy(logit, targets)
        text_loss = self.cross_entropy(logit.T, targets.T)
        loss = (image_loss + text_loss) / 2
        return loss

    def cross_entropy(self, x,  targets):
        logSoftmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * logSoftmax(x)).sum(1)
        return loss

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
class WordTokenizer(nn.Module):

    def __init__(self, input):
        super().__init__()

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.inputs = tokenizer(input, return_tensors="pt")
        

    def forward(self): # what it does on each call
        # return the token
        outputs = self.model(**self.inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states[:,0,:]
        

class ImageTokenizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        
    # input image
    # output vector
    def forward(self, input):
        return self.model(input)

"""
Projection class will allow us to project the images onto a 256
dim plane so we can accurately compare them
"""
class Projection(nn.Module):
    def __init__(self, embedding_dim, projection_dim=Config.projection_dim, dropout=Config.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU() # gaussian error linear units
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout) # prevents over-fitting
        self.layer_norm = nn.LayerNorm(projection_dim) # normalization

    def forward(self, input):
        projection = self.projection(input)
        input = self.gelu(projection)
        input = self.fc(input)
        input = self.dropout(input)
        input = input + projection
        input = self.layer_norm(input)
        return input

class Metric():

    def __init__():
        pass

    def update(value):
        pass

def make_loader(data, tokenizer): # inputs Dataset, outputs Dataloader
    transforms = get_transformers()
    dataset = CLIPModel(
        data["image"].values,
        data["labels"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.Dataloader(
        # configure upon parsing dataset
        dataset,
        batch_size=Config.batch_size
    )
    return dataloader

def train_epoch(model, dataloader): # plugs Dataloader through one iteration
    loss_meter = Metric()
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    for batch in tqdm_object:
        loss = model(batch)
        loss.backward()
        loss_meter.update(loss.item)
    return loss_meter

def valid_epoch(model, dataloader): # plugs Dataloader through one inference
    loss_meter = Metric()
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    for batch in tqdm_object:
        loss = model(batch)
        loss_meter.update(loss.item)
    return loss_meter

def make_training_df() : # creates training dfs and validation dfs
    df = pd.read_csv("dataset/labels.csv")
    max_id = df["id"].max() + 1
    image_ids = np.arrange(0, max_id)
    np.random.seed(420)
    
    test_ids = np.random.choice(
        image_ids, size=int(.2*len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in test_ids]
    
    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df

def main():
    train_df, valid_df = make_training_df() # returns dataframes for the training and validation
    train_loader = make_loader(train_df) # takes dataframe and returns dataloader
    valid_loader = make_loader(valid_df) # takes other dataframe and returnd dataloader
    model = CLIPModel().to(Config.device) # creates a CLIP model

    for epoch in range(Config.epochs): # iterates through as many epochs as needed
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader)
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss < best_loss: # saves current model if it is better than the last one
            best_loss = train_loss
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model")
