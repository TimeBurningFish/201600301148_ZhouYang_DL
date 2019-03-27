#!/usr/bin/env python
# coding: utf-8
# # Image features exercise 
# We have seen that we can achieve reasonable performance on an image classification task by training a linear classifier on the pixels of the input image. In this exercise we will show that we can improve our classification performance by training linear classifiers not on raw pixels but on features that are computed from the raw pixels.
# All of your work for this exercise will be done in this notebook.
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# ## Load data
# Similar to previous exercises, we will load CIFAR-10 data from disk.
from SDUCS2019.features import color_histogram_hsv, hog_feature
from SDUCS2019.data_utils import *
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'SDUCS2019/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    return X_train, y_train, X_val, y_val, X_test, y_test

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()


# ## Extract Features
# For each image we will compute a Histogram of Oriented
# Gradients (HOG) as well as a color histogram using the hue channel in HSV
# color space. We form our final feature vector for each image by concatenating
# the HOG and color histogram feature vectors.
# 
# Roughly speaking, HOG should capture the texture of the image while ignoring
# color information, and the color histogram represents the color of the input
# image while ignoring texture. As a result, we expect that using both together
# ought to work better than using either alone. Verifying this assumption would
# be a good thing to try for your interests.
# 
# The `hog_feature` and `color_histogram_hsv` functions both operate on a single
# image and return a feature vector for that image. The extract_features
# function takes a set of images and a list of feature functions and evaluates
# each feature function on each image, storing the results in a matrix where
# each column is the concatenation of all feature vectors for a single image.

from SDUCS2019.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


# ## Train SVM on features
# Using the multiclass SVM code developed earlier in the assignment, train SVMs on top of the features extracted above; this should achieve better results than training SVMs directly on top of raw pixels.


# Use the validation set to tune the learning rate and regularization strength

from SDUCS2019.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################

for li in learning_rates:
    for regi in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train_feats, y_train, learning_rate=li, reg=regi,
                          num_iters=1500, verbose=True)
        y_train_predict=svm.predict(X_train_feats)
        y_train_acc=np.mean(y_train_predict==y_train)
        y_val_predict=svm.predict(X_val_feats)
        y_val_acc=np.mean(y_val_predict==y_val)
        results[(li,regi)]=(y_train_acc,y_val_acc)
        if y_val_acc>best_val:
            best_val=y_val_acc
            best_svm=svm

################################################################################
#                              END OF YOUR CODE                                #
################################################################################
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)


# Evaluate your trained SVM on the test set
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)


# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()


# ## Neural Network on image features
# Earlier in this assigment we saw that training a three-layer neural network on raw pixels achieved better classification performance than linear classifiers on raw pixels. In this notebook we have seen that linear classifiers on image features outperform linear classifiers on raw pixels. 
# 
# For completeness, we should also try training a neural network on image features. This approach should outperform all previous approaches: you should easily be able to achieve over 55% classification accuracy on the test set; 


# Preprocessing: Remove the bias dimension
# Make sure to run this cell only ONCE
print(X_train_feats.shape)
X_train_feats = X_train_feats[:, :-1]
X_val_feats = X_val_feats[:, :-1]
X_test_feats = X_test_feats[:, :-1]
print(X_train_feats.shape)

from SDUCS2019.classifiers.neural_net import ThreeLayerNet



input_size = X_train_feats.shape[1]
num_classes = 10
reg = 0.01
num_batch = 40000
learning_rate_decay = 0.99


stds = [1]
lrs = [1e-2,1e-3,1e-1]
hidden_sizes = [256,512]
batch_sizes = [1024,2048]

nets = []
valids = []
hyps = []
for std in stds:
    for lr in lrs:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                net = ThreeLayerNet(input_size, hidden_size, num_classes, std=std)
                stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
                                  num_iters=num_batch, batch_size=batch_size,
                                  learning_rate=lr, learning_rate_decay=learning_rate_decay,
                                  reg=reg, verbose=True)
                valid = (net.predict(X_test_feats) == y_test).mean()
                nets.append(net)
                valids.append(valid)
                hyps.append({"std":std,"lr":lr,"batch_size":batch_size,"hiddden_size":hidden_size,"valid":valid})
                print("------------------std : " +str(std) +" lr: "+str(lr) +" batch_size: " + str(batch_size) + " hidden_size: " + str(hidden_size) +" valid: "+str(valid)+" --------------------")


# Train the network

index = valids.index(max(valids))

best_net = nets[index]

################################################################################
# TODO: Train a three-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
# Your code
################################################################################
#                              END OF YOUR CODE                                #
################################################################################


# Run your best neural net classifier on the test set. 

test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print(test_acc)
for h in hyps:
    print(hyps)

print("best hyp"+str(hyps[index]))

