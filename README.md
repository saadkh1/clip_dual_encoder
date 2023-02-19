# Visual and Vision-Language Representation Pre-Training with Contrastive Learning

In this repository, I present my approach as a family of vision-language foundation systems. These systems, which are considered among the most advanced in the field of artificial intelligence, are used to solve a variety of important tasks, such as generation, retrieval, and classification tasks. The basis of the vision-language systems is a combination of pre-trained encoder models of computer vision and natural language processing.

## Setup

I used Python 3.7 and [PyTorch](https://pytorch.org/) 1.7.1 to train and test my models. It also requires the installation of timm and transformers packages with the following versions:

```bash
$ pip install timm==0.6.7
$ pip install transformers

## Training the model:
```
python main.py
```

During this research, I trained our models on several image encoder models, such as deit3, efficientnet_b8, convnext, swinv2, and fbnetv3, as well as text encoder models, such as roberta, xlm-roberta, xlnet, albert, electra, and bert, on a small-scale dataset. I concluded that changing image and text encoder models would necessarily change the efficiency of the model. I was able to show that encoder models like ConvNeXt and RoBERTa produce better results than more popular encoder models like BERT and ViT. I also found that training our model on a small-scale data set produced accurate results.

Below in this table are the best results I got after testing more than 80 pairs of image and text encoders in 20 epochs.

|  Image and Text Encoders  | Top-1 accuracy | Top-5 accuracy | 
|:-------------------:|:----------:|:----------:|
|  Swin + BERT        |    13.99    |    31.65    |
|  Swin + ALBERT      |    13.28    |    29.04    |
| Swin + RoBERTa      |    14.83    |    32.64    |
| ConvNeXt + BERT     |    16.63    |    35.31    |
| ConvNeXt + ALBERT   |    13.46    |    29.46    |
| ConvNeXt + RoBERTa  |    17.31    |    35.31    |

The model composed by the ConvNeXt and RoBERTa encoders has achieved satisfactory results on image-text retrieval tasks on some datasets, such as the ImageNetV2 dataset, the Unsplash dataset, and more other datasets, using different queries (textual queries, visual queries, and visual + textual queries) as shown in these notebooks [Search_In_Unsplash.ipynb](https://github.com/saadkh1/clip_dual_encoder/blob/main/Search_In_Unsplash.ipynb) and [Image_To_Text_Search.ipynb](https://github.com/saadkh1/clip_dual_encoder/blob/main/Image_To_Text_Search.ipynb).It also achieved these results on the video-text retrieval task for videos from YouTube [Test_video.ipynb](https://github.com/saadkh1/clip_dual_encoder/blob/main/Test_video.ipynb).This model has outperformed many other models in image classification tasks on some datasets, such as Food-101, CIFAR-10, CIFAR-100, Describable Textures (DTD), Oxford-IIIT Pets (Pets), MNIST, STL-10, the German Traffic Sign Recognition Benchmark (GTSRB), and Rendered SST2 (SST) [CR_classification.ipynb](https://github.com/saadkh1/clip_dual_encoder/blob/main/CR_classification.ipynb).Also, I tested this model on zero-shot image classification tasks as it showed its efficiency in some datasets, such as the (CIFAR-100 classes with some random images), the (puppy and bagel), and the (chihuahua and muffin) datasets [Zero-Shot Image Classification.ipynb](https://github.com/saadkh1/clip_dual_encoder/blob/main/Zero-Shot%20Image%20Classification.ipynb).

I did this by applying the contrastive learning method to the image-text pair dataset, which was used to train my models. Then I used the Zero-Shot method to retrieve images or texts. My model achieved all of these results, considering the size of the Flickr30k dataset and the number of epochs used in training.
