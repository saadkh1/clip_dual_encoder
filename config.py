import torch

debug = True
image_path = "C:/saadkh/flickr-image-dataset/flickr30k_images/flickr30k_images"
captions_path = "C:/saadkh/flickr-image-dataset/flickr30k_images"
batch_size = 32
num_workers = 2
lr = 1e-3
weight_decay = 1e-3
image_encoder_lr = 1e-5
text_encoder_lr = 1e-5
patience = 1
factor = 0.8
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'convnext_base_in22ft1k'
image_embedding = 1024
text_encoder_model = "roberta-base"
text_embedding = 768
text_tokenizer = "roberta-base"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1