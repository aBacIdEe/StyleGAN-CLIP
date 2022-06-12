import cv2
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A # provides fast image augmentation and implements image transform
import timm
from transformers import DistilBertTokenizer, DistilBertModel

import torch
from torch import nn
import torch.nn.functional as F

# Classes we'll probably need

class Config:
    image_path = "Datasets/flickr30k_images"
    captions_path = "Datasets/results.csv"
    max_length = 32
    size = 256 # TODO check how big are images
    projection_dim = 256 # projection dimension size
    temperature = 1 # confidence
    image_embedding = 2048
    text_embedding = 768
    batch_size = 32
    epochs = 8 # epochs to train for
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.1
    image_encoder_lr = 0.00001
    text_encoder_lr = 0.00001
    head_lr = 0.0001
    weight_decay = 0.0001

class Dataset(torch.utils.data.Dataset):

    def __init__(self, files, captions, tokenizer, transforms):
        self.files = files
        self.captions = list(captions)

            
        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=Config.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{Config.image_path}/{self.files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, dsize=(Config.size, Config.size), interpolation=cv2.INTER_CUBIC)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)

def get_transformers():
    return A.Compose(
        [
            A.Resize(Config.size, Config.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True)
        ]
    ) 

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
        image_loss = self.cross_entropy(logit.T, targets.T)
        text_loss = self.cross_entropy(logit, targets)
        loss = (image_loss + text_loss) / 2
        return loss.mean()

    def cross_entropy(self, x,  targets):
        logSoftmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * logSoftmax(x)).sum(1)
        return loss

class WordTokenizer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)


    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, 0, :]
        

class ImageTokenizer(nn.Module):

    def __init__(self):
        super().__init__()
        # self.model = models.resnet50(pretrained=True)
        self.model = timm.create_model(
            "resnet50", True, num_classes=0, global_pool="avg"
        )
        
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

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3
    
    def update(self, val, count=1):
        self.count += 1
        self.sum += val*count
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.avg:.4f}"

def make_loader(data, tokenizer): # inputs Dataset, outputs Dataloader
    transforms = get_transformers()
    dataset = Dataset(
        data["image_name"].values,
        data["comment"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        # configure upon parsing dataset
        dataset,
        batch_size=Config.batch_size
    )
    return dataloader

def train_epoch(model, dataloader, optimizer): # plugs Dataloader through one iteration
    loss_meter = Metric()
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    for batch in tqdm_object:
        batch = {k: v.to(Config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # updates weights
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
    return loss_meter

def valid_epoch(model, dataloader): # plugs Dataloader through one inference
    loss_meter = Metric() # running total of loss
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    for batch in tqdm_object:
        batch = {k: v.to(Config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
    return loss_meter

def make_training_df() : # creates training dfs and validation dfs
    df = pd.read_csv(Config.captions_path, delimiter='|')
    # print(str(df))

    max_id = len(df.index)
    image_ids = np.arange(0, max_id)
    np.random.seed(420)
    
    test_ids = np.random.choice(
        image_ids, size=int(.2*len(image_ids)), replace=False # validation ids are randomly chosen
    )
    train_ids = [id_ for id_ in image_ids if id_ not in test_ids] # training ids are everything except the validation ids
    # abc = list(df["comment"].values) # something was wrong with the dataset
    # for i in train_ids:
    #     if abc[i] != str(abc[i]):
    #         abcd = 1
    train_df = df.iloc[train_ids].reset_index(drop=True)
    test_df = df.iloc[test_ids].reset_index(drop=True)
    return train_df, test_df

def main():
    train_df, valid_df = make_training_df() # returns dataframes for the training and validation
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # print(train_df.keys)
    train_loader = make_loader(train_df, tokenizer) # takes dataframe and returns dataloader
    valid_loader = make_loader(valid_df, tokenizer) # takes other dataframe and returnd dataloader
    model = CLIPModel().to(Config.device) # creates a CLIP model
    # model.cuda()
    params = [
        {"params": model.image_encoder.parameters(), "lr": Config.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": Config.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": Config.head_lr, "weight_decay": Config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0) # creates the optimizer object

    best_loss = float('inf')
    for epoch in range(Config.epochs): # iterates through as many epochs as needed
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer) # trains an epoch
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader) # validates an epoch
        
        if valid_loss.avg < best_loss: # saves current model if it is better than the last one using valid_loss
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model")

# main()