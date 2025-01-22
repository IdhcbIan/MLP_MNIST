# MNIST Classification with PyTorch and AWS SageMaker deployment

## Overview

This project demonstrates image classification on the MNIST dataset using PyTorch, a popular deep learning framework.

### What is MNIST?

MNIST (Modified National Institute of Standards and Technology) is a widely-used dataset to test machine learning models.
The problem consists of 28x28 images of handwritten digits from 0 to 9, with the goal of classifying the digit in the image.


## AWS Cloud Inference

- This project uses AWS SageMaker to deploy a PyTorch model to an API endpoint. While the model is simple, the whole point of the project is to learn how to deploy a model to production and connect to a user friendly interface in a react app.

- To try it out go to my lab:(It does take a few seconds to load the first example due to the type of ec2 instance I am using)

- https://iansmainframe.com/lab


## Requirements

- PyTorch
- numpy
