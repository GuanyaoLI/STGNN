# STGNN -DJD (Spatial-Temporal Graph Neural Network considering Dynamic and Joint Dependency)

## About
Implementation of the paper [A Data-Driven Spatial-Temporal Graph Neural Network for Docked Bike Prediction].

Thanks for your attention to this work. More details will be released soon.

## Installation
Requirements

 - Python 3.7 (Recommend Anaconda)
 - pytorch 1.8.1

## Description

  - The Chicago and LA datasets are Chicago.zip and la.zip. Please unzip them and put in the folders of '/data/chicago' and '/data/la/'before training and testing the model.
  - Run with "python train.py" to train the model and "python test.py" to test the model.

## Discussion

A simple way to extend our approach for multiple slot prediction is replacing the model output $\{O_t, I_t\}$ as $\{O_t, \cdots O_{t+k}, {I_t, \cdots, I_{t+k}}\}$ in both training and prediction phases. We will study as a future work more sophisticated approaches for multi-step prediction considering dynamic and joint spatial-temporal dependency.
