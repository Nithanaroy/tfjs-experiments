# CNN Training and Inference on MNIST 

In this submodule, MNIST binary dataset is downloaded from its creator's site, parsed and trained completely in JavaScript. This data processing script is the unique piece of the module and can be found in [data.js](data.js) file.This is a much simpler but equally feature rich implementation than the [one](https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%201%20-%20TensorFlow-JS/Week%202/Examples/data.js) from [Browser based models with Tensorflow.js](https://www.coursera.org/learn/browser-based-models-tensorflow/home/welcome) course.

This is tested in Chrome 79

## Setup
- Download MNIST training data images and labels from http://yann.lecun.com/exdb/mnist/, unzip and put in ./data/ folder
- Serve this folder behind a simple web server. I used `python3 -m http.server 8000` after installing Python 3.6.
