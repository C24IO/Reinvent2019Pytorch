# Reinvent2019Pytorch

## Build, Test, and Tune Machine Learning Models with PyTorch

![](./img/SMPT.png)

PyTorch is a deep learning framework that is becoming popular especially for rapid prototyping of new models. You can get started easily with PyTorch using Amazon SageMaker, a fully managed service to build, train, and deploy machine learning models at scale. In this workshop, we build a natural language processing model to analyze text. 

NLP solutions based on PyTroch have several principle components, text preprocessing, text representation using techniques such 
as GloVe: Global Vectors for Word Representation, state of the art general purpose, pre-trained models such as 
Bidirectional Encoder Representations from Transformers or BERT. 
In addition to these principle components, we need to plan for a full end-to-end model training and deployment lifecycle. 
This lab aims to prepare you for developing end-to-end NLP solutions using [PyTorch](pytorch.org) for the principle components 
and [Amazon SageMaker](https://aws.amazon.com/sagemaker/) for the full Machine Learning Lifecycle.

Each lab is independent so you can pickup each as your own choosing. We will be going through this serially to facilitate explanations.
We wish to thank the authors and code committers of projects listed in the reference section, the labs would not be possible without this 
excellent material. Pull requests for improvements and additions most welcome.    
   
### To run this workshop locally on your laptop -

```bash
$ cd src
$ ./boot.sh
$ source venv/bin/activate
``` 
![](./img/ml-concepts-10.png)

## Lab 1

In lab 1 we exercise all the various components of SageMaker that have native Pytorch support.
We start with a notebook that has a PyTorch kernel. Then we use sagemaker.pytorch.estimator.PyTorch to train 
a multi-layer LSTM RNN model. This model predicts the probability of the next word in the sequence, based on the words 
already observed in the sequence. We later in the notebook use this trained model to generate a legible text from
a random seed. At each stage of the model we use SageMaker to train, create model and finally deploy 
the model as a real time endpoint. 


## Lab 2 

In lab 2 we use SageMaker to fine tune Hugging Face's pre-trained BERT based on PyTorch. We use FastBert framework for this, we fine-tune
BERT for multi-class classification of comments. This lab is based on a Kaggle competition that is aimed at 
automatically labelling comments to be in one or more of the six possible types of comments - 
toxic, severetoxic, obscene, threat, insult, identityhate. 

We start off with first create a container, use SageMaker to train this container on a corpus of comments data.
Finally we deploy a real time endpoint and run it on a set of comments. 

## Lab 3

TBD 

## Lab 4

In lab 4 we work with newly released Captum.ai. In this lab we load a CNN model that has been pre-trained on IMDB dataset 
for sentiment analysis. Then we use this model to detect sentiments of each statement before using the integrated 
gradients method to interpret those predictions. 


## Extras

We have and will continue to add a few extras to this set of labs. *bonus* is a directory that contains some helper scripts. *impatient* contains the solved labs with outputs.


## References & Source Material
* [SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples)
* [Introducing FastBert — A simple Deep Learning library for BERT Models](https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384)
* [Kaushal Trivedi's FastBert](https://github.com/kaushaltrivedi/fast-bert)
* [🤗 Hugging Face - Solving NLP, one commit at a time!](https://github.com/huggingface)
* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
* [Kaggle Competetion - Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
* [Jeffrey Hsu, Susannah Klanecek: A Deep Dive into NLP with PyTorch | PyData London 2019](https://www.youtube.com/watch?v=Apx_1erbQB4)
* [A Deep Dive into NLP with PyTorch](https://github.com/scoutbee/pytorch-nlp-notebooks)
* [A Deep Dive into NLP with PyTorch - Slides](https://docs.google.com/presentation/d/1zyuwCx7knqnP-LJswlDfWSmk5FhFgFmYJGqdEZn8yhc/edit#slide=id.g33c734b530_0_656)
* [Captum - Interpreting text models: IMDB sentiment analysis](https://captum.ai/tutorials/IMDB_TorchText_Interpret)
* [Ben Trevett's - PyTorch Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
* [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
 


