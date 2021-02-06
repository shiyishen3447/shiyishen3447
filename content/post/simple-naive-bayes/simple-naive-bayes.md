---
title: Simple Naive Base Classifier with Movie Reviews
subtitle: This is the most ____ movie I have ever watched!
summary: This is the most ____ movie I have ever watched!
authors:
tags: []
categories: []
date: 2019-07-12
math: true
diagram: true
featured: true
image:
  filename: ""
  focal_point: ""
  preview_only: true
  caption: ""
  alt_text: ""
---

```python
from IPython.core.display import Image
Image("/Users/shiyishen/GitHub/shiyishen3447/content/post/simple-naive-bayes/featured.png")
```
 
![png](simple-naive-bayes_files/simple-naive-bayes_1_0.png)
    



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
