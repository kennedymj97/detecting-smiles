# Detecting Smiles

## Summary

Project that detects whether or not a person is smiling using Deep Learning in real time through a webcam feed.

## Key Learnings

 - Python (Pytorch, FastAI, Caffe2, onnx)
 - Deep learning architectures for Computer Vision
 - Finding and processing data
 - Use of Google Cloud Platform

## Example

![Image showing software demonstration](https://i.imgur.com/faS7yxc.png)

## Development Process

### Challenges

- Finding suitable training data.
- Finding faces and carrying out carrying out detection in real time.
- Working out best Deep Learning architecture and training and using the model.

### Finding dataset

- The dataset used to train the model is the GENKI-4K dataset.
- The “GENKI-4K subset contains 4000 face images labeled as either “smiling” or “non-smiling””.
- The “GENKI Database is an expanding database of images containing faces spanning a wide range of illumination conditions, geographical locations, personal identity, and ethnicity.”

![Example images from the dataset](https://i.imgur.com/CdzdAha.png)

### Traning the models

The FastAI library was used to train the models, see train_models.ipynb for the training. The following are the steps carried out to train the model:

1. Data Augmentation
    1. Transforms were applied to the dataset to create a more diverse dataset.
    1. Allows for worse image conditions and helps to reduce overfitting.
1. Training final layer
    1. A final layer was added to the ResNet18 model and trained to classify between images of people smiling and not smiling.
1. Fine-tuning earlier layers
    1. All the layers in the ResNet18 model “unfrozen” and trained to tweak the activations to this application.
1. Convert model to onnx
    1. Model was converted to an onnx format so it could be loaded as a Caffe2 model for faster prediction.

### Testing the speed of the models

To ensure that the model used could run on the cpu in real time I trained a variety of models and wrote a jupyter notebook that would carry out 1000 predictions using the models and then calculate the average time for prediction and plot a graph.

![Deep Learning model speed test results](https://i.imgur.com/Rtpepoo.png)

These tests showed (as expected) that exporting the models from PyTorch and running them through Caffe2 was by far the quickest method. Also, the results showed that the depth of the network greatly affected the prediction time. For this reason the ResNet18 model was chosen.

See speed_test.ipynb for implementation.

### Writing the main programme

The main challenge when writing the main programme was to ensure as high frame rate as possible, this was carried out using the following methods:

- The faces were detected using Haar Cascades which is not ideal but is necessary to maintain high framerates.
- Stack all of the faces in each frame together so they can be put through the model as one batch.
- Use the multiprocessing library so the main loop is not dependant on prediction time.

![Image of part of the main programme](https://i.imgur.com/C1cmcIZ.png)
