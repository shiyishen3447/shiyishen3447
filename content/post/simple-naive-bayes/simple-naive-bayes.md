---
title: Simple Naive Base Classifier with Movie Reviews
subtitle: This is the most ____ movie I have ever watched!
date: 2021-02-06T19:37:06.550Z
summary: This is the most ____ movie I have ever watched!
draft: true
featured: true
authors: null
math: true
diagram: true
tags: []
categories: []
image:
  filename: featured.jpg
  focal_point: Smart
  preview_only: true
  caption: ""
  alt_text: ""
---
```python
from IPython.core.display import Image
Image("/Users/shiyishen/GitHub/shiyishen3447/content/post/simple-naive-bayes/featured.png")
```

​
​    

## Numpy and Defaultdict for Feature Vectors

Here we will import `numpy` and build matrices and arrays to support our data structure 

In the past, I have primarily used dictionary for data storage. Alternatively, `numpy` can do powerful magic tricks to 
the arrays and matrices. Therefore, in this project I use `numpy` to store all the `feature vectors`.

```python
import os 
import numpy as np
from collections import defaultdict
import nltk 
#nltk.download(punkt)
```