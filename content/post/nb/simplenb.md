---
title: A Simple Naive Bayes Movie Review Classifier
subtitle: Classify movie reviews with a generative model
summary: Classify movie reviews with a generative model
authors:
tags: []
categories: []
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: featured.png
  focal_point: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
## Is It An Action or A Comedy Film?

This notebook is a step by step walk-through on how to train a simple naive bayes classfier to recognize the genre of the film through its reivew. 

Before we start, below is a picture demonstration of the equation for calculating the likelihood.
1. Prior: number of files in given class, i.e. if 2 out of 5 reviews are action films, 0.4 will be its prior prob.
2. Likelihood or P(feature|class): give num of features, what's the likelihood that its an action film ((i.e. fly,fun,kick,hit)|action)
3. Evidence: number of data points (here namely reviews) we have. 
$$P(A \mid B) = \frac{ P( B | A )P( A ) }{P(B)} = {{\sum}}{ P( B | A )P( A ) }$$
Note that in this exercise about computing the denominator for the naive Bayes classifier, we can ignore the denominator since we're comparing P(action | review) and P(comedy | review) and so can cancel out their denominators to simplify our work.

## Building and Storing Feature Vectors

Create parameters to store the **features** into an appropriate data structure of your choice. 

Here `numpy` is used to create matrices for creating **feature vectors**
In the past, I have primarily used `dictionaries` for storing data. Alternatively, `numpy` supports various magic operations on the data structure and is very powerful. Therefore, here `numpy` is used.

Please click for more information about how to use [numpy](https://cs231n.github.io/python-numpy-tutorial/).




```python
import os
import numpy as np
from collections import defaultdict
import nltk
# nltk.download()
# nltk.download('punkt')


prior = np.zeros(2)       #self.prior
N_doc = 0 #number of documents
N_class = np.zeros(2) #
doc_action = []
doc_comedy = []
doc_all = []
```

Before we can do anything, we first have to load data into our workspace.
The following code scrapes all the roots, dirs, and directories for the files we need


```python
for root, dirs, files in os.walk("Users/shiyishen/doc/class_material/COSI_114_FoCL/homework/PA2/⁨movie_reviews_small/train"):
    for name in files:
        N_doc += 1 #num of documents 
        with open(os.path.join(root, name)) as f:
            text = nltk.word_tokenize(f.read())
        if root == r"Users/shiyishen/doc/class_material/COSI_114_FoCL/homework/PA2/movie_reviews_small⁩/train/action":
            N_class[0] += 1
            doc_action.extend(text)
        else:
            N_class[1] += 1
            doc_comedy.extend(text)
        doc_all.extend(text)
```

Some basic text tokenization and counting 


```python
dict_action = nltk.FreqDist(doc_action)      #bigdoc[action]
dict_comedy = nltk.FreqDist(doc_comedy)      #bigdoc[comedy]
dict_voc = nltk.FreqDist(doc_all)            #vocabulary
```

Remember we have defined three parameters `dict_action`, `dict_comedy`, and `doc_all` in the previous cell. As we have iterated through the data folder and loaded in their corresponding text, what's left is count the number of words that each category and the overall data contain. We'll use NLTK's `.FreqDist` to directly compute their **frequency distribution**.

## Training Our NB Classifier 
Now we can then start to create our **feature vectors**. We do it first by creating an array to store all of our features, which are unique words in our training files. Or you could do your own feature selection. 


```python
N_features = len(dict_voc.keys())
likelihood = np.zeros((2, N_features)) #for storing all the likelihood for each feature
class_dict = ['action', 'comedy']
features = []
for i in dict_voc.keys():
    feature_dict.append(i)
```


Now we are done with data prep and preprocessing. Let's head into training our model 

If you still remember the equation. To calculate the NB distribution of a given class, we need its **prior probability** and the **likelikhood** that each feature appears in the class. 
Here we use log pace to smoothe our calculation, as we might encounter some significantly small number.

We use **Laplace's smoothing** technique also called **add-one smoothing**.


```python
$$ 
```


```python
log_prior = np.zeros(len(prior))
log_likelihood = None 

for i in range(2):
    log_prior[i] = np.log(N_class[i]/N_doc)

for i in range(len(class_dict)):
    count_wc = np.zeros(len(features))
    sum_count = len(features)  
    if i == 0:  
        for j in range(len(features)):

            if features[j] in dict_action.keys():
                count_wc[j] = dict_action[features[j]]
                sum_count += count_wc[j] 
            else:
                count_wc[j] = 0
        for j in range(len(features)):
            likelihood[0][j] = (count_wc[j] + 1) / sum_count
    if i == 1: 
        for j in range(len(features)):

            if features[j] in dict_comedy.keys():
                count_wc[j] = dict_comedy[features[j]]
                sum_count += count_wc[j]
            else:
                count_wc[j] = 0
        for j in range(len(features)):
            likelihood[1][j] = (count_wc[j] + 1) / sum_count
log_likelihood = np.log(likelihood)
```

## Predict and Classify




```python
results = defaultdict(dict)
for root, dirs, files in os.walk(dev_set):
    for name in files:
        #print(name)
        if name!='.DS_Store':
            feature_vector = np.zeros(len(self.feature_dict))
            results[name] = []
            with open(os.path.join(root, name)) as f:
                text = nltk.word_tokenize(f.read())
                #print(text)
                if root == '/Users/shiyishen/Downloads/movie_reviews/dev/pos':
                    results[name].append('pos')
                else:
                    results[name].append('neg')

                dict_text = nltk.FreqDist(text)
                for i in range(len(self.feature_dict)):
                    if [i] in text:
                        feature_vector[i] = dict_text[self.feature_dict[i]]
                    else:
                        pass

            feature_vector.transpose()
            compare = np.dot(self.loglikelihood, feature_vector)
            compare = compare + self.logprior
            if compare[0] > compare[1]:
                results[name].append('pos')
            else:
                results[name].append('neg')
            # create feature vectors for each document
                pass
        # get most likely class
```


```python

```
