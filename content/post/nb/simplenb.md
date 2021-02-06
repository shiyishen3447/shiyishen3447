---
title:A Simple Naive Bayes Movie Review Classifier
subtitle: Classify movie reviews with a generative model
summary: Classify movie reviews with a generative model
authors: null
date: 2019-02-05T00:00:00Z
lastMod: 2019-09-05T00:00:00Z
tags: []
categories: []
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
projects: []
image:
  filename: simplenb_2_0.png
  focal_point: Smart
  preview_only: false
---
## Is It An Action Film or A Comedy?

This notebook demonstrates how to train a simple naive bayes classifier to recognize the genre of the film through its review. 

Before we start, below is a picture demonstration of the equation for calculating the likelihood.

1. Prior: number of files in given class, i.e. if 2 out of 5 reviews are action films, 0.4 will be its prior prob.
2. Likelihood or P(feature|class): give num of features, what's the likelihood that its an action film (words|action)
3. Evidence: number of data points (here namely reviews) we have. 

Note that in this exercise about computing the denominator for the naive Bayes classifier, we can ignore the denominator since we're comparing P(action | review) and P(comedy | review) and so can cancel out their denominators to simplify our work.

```python
from IPython.core.display import Image
Image('https://javafreecode.files.wordpress.com/2015/02/posterior-full.png')
```  
![Simplenb 2 0](simplenb_2_0.png)

## Building and Storing Feature Vectors

Create parameters to store the **features** into an appropriate data structure of your choice. 

We will have to build our own **feature vectors** using tools from NLTK
In the past, I have primarily used `dictionaries` for storing features. Alternatively, `numpy` supports various magic operations on the data structure and is very powerful. Therefore, for this project `numpy` is used to build our feature vectors.

See more information on how to use [numpy](https://cs231n.github.io/python-numpy-tutorial/) for feature engineering.

```python
import os
import numpy as np
from collections import defaultdict
import nltk
#nltk.download('punkt')

prior = np.zeros(2)       #self.prior
N_doc = 0 #number of documents
N_class = np.zeros(2) #
doc_action = []
doc_comedy = []
doc_all = []
```

## Edit your post metadata

The first cell of your Jupyter notebook will contain your post metadata ([front matter](https://sourcethemes.com/academic/docs/front-matter/)).

In Jupyter, choose *Markdown* as the type of the first cell and wrap your Academic metadata in three dashes, indicating that it is YAML front matter: 

```
---
title: My post's title
date: 2019-09-01

# Put any other Academic metadata here...
---
```

Edit the metadata of your post, using the [documentation](https://sourcethemes.com/academic/docs/managing-content) as a guide to the available options.

To set a [featured image](https://sourcethemes.com/academic/docs/managing-content/#featured-image), place an image named `featured` into your post's folder.

For other tips, such as using math, see the guide on [writing content with Academic](https://sourcethemes.com/academic/docs/writing-markdown-latex/). 

## Convert notebook to Markdown

```bash
jupyter nbconvert index.ipynb --to markdown --NbConvertApp.output_files_dir=.
```

## Example

This post was created with Jupyter. The orginal files can be found at https://github.com/gcushen/hugo-academic/tree/master/exampleSite/content/post/jupyter