Capstone Proposal
==================
Baptiste Azema <br>
December 4th, 2020

# Pokémon classifier

## Domain Background
Pokémon are creatures that inhabit the fictional Pokémon World. Each creature has unique design and skills.
We will focus on the [Generation One](https://en.wikipedia.org/wiki/List_of_generation_I_Pok%C3%A9mon) featuring
the original 151 fictional species of creatures introduced in the 1996 Game-Boy games "Pokémon".

As our world is becoming more and more bizarre, a future were Pokémon and Humans coexist might be possible. 
We might need some tool to identify if a photo shows a Pokémon, a human, and which Pokemon it looks like.

## Problem Statement

This project aims to classify images as Pokémon species. 
When the input image is a Pokémon, the algorithm will respond the Pokémon name.
When the input image is a human face or anything else, the algorithm will respond if it's human and 
which Pokémon it looks like.

Inspired by udacity's [CNN Project: Dog Breed Classifier](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)

## Datasets
    
- [Pokemon Generation One](https://www.kaggle.com/thedagger/pokemon-generation-one). 
  This dataset will be used to train and evaluate a classifier of Pokémon species.
- [Labeled Faces in the Wild Home](http://vis-www.cs.umass.edu/lfw/lfw.tgz).
  This dataset will be used to evaluate a human face detector.


## Solution Statement

The first thing to do will be to detect if the image represents a Pokemon, a Human face, or anything else. The Python
 library "OpenCV" might do the job to detect humans, but a machine learning model is needed to detect if it's a Pokemon or not. 

The next step will be to classify the Pokemon/Human/else in the image.
This second step can be done using a custom CNN model built from scratch using PyTorch, 
or using transfer learning with pre-trained models, like "ResNet-50" or "InceptionV3".

We will use AWS SageMaker to benefit from the GPU compute and to industrialize our application.

## Benchmark Model

Many image classifications model are out there, for instance [ResNet](https://arxiv.org/abs/1512.03385). It uses the 
state-of-the-art Convolutional Neural Networks (CNN) models for classification.

## Evaluation Metrics

Evaluation will be done using a "test dataset" not seen by the model but on which we have the "Ground truth".
We will run the model on the dataset and compare the result with the "Ground truth" using the accuracy score. 

## Project Design

- Build a Human detector using OpenCV
- Train a Pokemon detector using transfer learning with PyTorch on AWS Sagemaker.
- Train a Pokemon classifier using transfer learning with PyTorch on AWS Sagemaker. 
  I'm planning to build a classifier from scratch and then uses transfer learning with 2 pre-trained models to compare the performances.

Then, a python program will:
 - take an image as input
 - use the Pokemon/Human detector to detect if it's a Pokemon, a Human, or something else
 - use the trained Pokemon classifier to give the name of the Pokémon (if it's a Pokémon), 
   or the Pokémon it looks like (if it's a Human or something else)

I would love to deploy this application on Pypi.org and make it available to the world.
