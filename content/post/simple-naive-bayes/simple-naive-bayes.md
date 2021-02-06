---
title: Simple Naive Base Classifier with Movie Reviews
subtitle: This is the most ____ movie I have ever watched!
summary: This is the most ____ movie I have ever watched!
authors:
- admin
tags: []
categories: []
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

```python
from IPython.core.display import Image
Image("/Users/shiyishen/GitHub/shiyishen3447/content/post/simple-naive-bayes/featured.png")
```




    
![png](./simple-naive-bayes_1_0.png)
    



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