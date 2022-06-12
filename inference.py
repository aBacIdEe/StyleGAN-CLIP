# Uses the trained model to predict stuff

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

from train import CLIPModel, make_loader, make_training_df

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

'''
Gets a set of data to inference off of, and find their image embeddings to compare to.
'''

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    valid_loader = make_loader(valid_df, tokenizer)
    
    model = CLIPModel().to(Config.device)
    model.load_state_dict(torch.load(model_path, map_location=Config.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(Config.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

_, valid_df = make_training_df()
model, image_embeddings = get_image_embeddings(valid_df, "trained.pt")

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(Config.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{Config.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()

find_matches(model, 
             image_embeddings,
             query="maxwell",
             image_filenames=valid_df['image_name'].values,
             n=9)