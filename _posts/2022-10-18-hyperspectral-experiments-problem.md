---
layout: post
title: Experimental Design Issues in Hyperspectral Image Classification
usemathjax: true
---

Hyperspectral images are optical images that consist of tens or hundreds of spectral bands. Therefore, they can carry much more information than RGB images.

Classification of hyperspectral images has been studied for years, and most of the works in the literature take a contextual approach, so they use both spectral information and pixel neighborhood relations.

Preparing a hyperspectral remote sensing dataset is not easy. You can get a hyperspectral camera, put it on a plane, drone, or a satellite, and gather your data. However, labeling these is the real problem. Some experts have to go to the site and check out the landcover material type, whatever it is. And it is very costly, difficult, and sometimes impossible. If you are conducting research and hoping to use publicly available datasets, your options are limited. The available datasets generally consist of a single image in which you need to do training, validation, and tests. Data splitting is where the problem occurs. If you take a portion of the image, you cannot get data from most of the classes as your training or test samples. If you do random splitting, applying spatial models is difficult. And some classes have a very small number of pixels. 

Like almost every other classification problem, deep learning has been used for spectral-spatial classification of hyperspectral images. If you do random splits, and use a CNN, the pixel in the center of the filter can be in the training set, but its neighbors can be from the test set, and you still use them in the training. Which causes an information leakage. The same thing happens for the test set, too.

There are thousans of papers applying models like CNNs or conditional random field, without any explanation of their solution to this problem. When they do not care about this, they get over-optimistic results. It is also very hard to convince reviewers about this, they expect you to use 2D/3D models, and get better results than those papers with bad experimental design. It is really weird that there are even highly cited papers from "celebrity" scientists which either do not explain their strategy or just do not care about it.
