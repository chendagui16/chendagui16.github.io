---
title: NLP3 More Word Vectors
mathjax: true
date: 2017-04-16 15:52:42
categories: NLP
tags: [machine learning, NLP]
---
reference
[cs224n](http://web.stanford.edu/class/cs224n/syllabus.html)
# lecture 3 More Word Vectors
## Stochastic gradients with word vectors
* But in each window, we only have at most $2m+1$ words, so $\nabla\_\theta J\_t(\theta)$ is very sparse!
* We may as well only update the word vectors that actually appear!
* **Solution**: either you need sparse matrix update operations to only update columns of full embedding matrices $U$ and $V$, or you need to keep around a hash for word vectors
* If you have millions of word vectors and do distributed computing, it is important to not have to send gigantic updates around!

### Approximations
* The normalization factor is too computationally expensive
$$ p(o|c) = \frac{\exp (u\_o^T v\_c)}{\sum\_{w=1}^V \exp (u\_w^T v\_c)} $$
* Implement the skip-gram model with **negative sampling**
* Main idea: train binary logistic regressions for a true pair (center word and word in its context window) versus a couple of noise pairs (the center word paired with a random word)

**The skip-gram model and negative sampling**
* From paper _Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al. 2013)_
* Overall objective function: $J(\theta) = \frac{1}{T} \sum\_{t=1}^{T} J\_t(\theta)$
$$J\_t(\theta) = \log \sigma(u\_o^T v\_c) + \sum\_{i=1}^{k} E\_{j\sim P(\omega)} \left[ \log \sigma (-u\_j^T v\_c)\right] $$
* Where $k$ is the number of negative samples and we use
* $\sigma$ is sigmoid function
* So we maximize the probability of two words co-occurring in first log

## Negative sampling
word2vec is a **huge** neural network!
The author of Word2Vec addressed the issue in their second [paper](https://arxiv.org/pdf/1310.4546.pdf)
There are **three** innovations in this second paper:
1. Treating common word pairs or phrases as a single "words" in their model
1. Sub-sampling frequent words to decrease the number of training examples
1. Modifying the optimization objective with a technique they called "Negative Sampling", which causes each training sample to update only a small percentage of the model's weights

**Note**: Sub-sampling frequent words and applying Negative Sampling not only reduced the compute burden of the training process, but also improved the quality of their resulting word vectors as well.

### Word Pair and "Phrases"
Example: a word pair like "Boston Globe" has a much different meaning than the individual words "Boston" and "Globe". So it makes sense to treat "Boston Globe" as a single word

**Method of phrase detection**: it is covered in the "Learning Phrases" section of [paper](http://arxiv.org/pdf/1310.4546.pdf). And the code is available in _word2phrase.c_ of their published [code](https://code.google.com/archive/p/word2vec/)

### Sub-sampling Frequent Words
As this example
![example](http://i4.buimg.com/567571/4ec276dc62e7c4c8.png)

There are two problems with common words like _the_:
1. When looking at word pairs, ( _fox_, _the_ ) doesn't tell use much about the meaning of _fox_. _the_ appears in the context of pretty much every word.
1. We will have many more samples of ( _the_, $\dots$) than we need to learn a good vector for _the_.

> Word2Vec implements a "sub-sampling" scheme to address this. For each word we encounter in training text, there is a chance that we will effectively delete it from the text. The probability that we cut the word is related to the word's frequency

If we have a window size of 10, and we remove a specific instance of _the from our text_:
1. As we train on the remaining words, _the_ will not appear in any of their context windows.
1. We'll have 10 fewer training samples where _the_ is the input word.

**Sampling rate**
For any word $w\_i$, $z(w\_i)$ is the fraction of the total words in the corpus that are that word.
$P(w\_i)$ is the probability of _keeping_ the word:
$$ P(w\_i) = \left( \sqrt{\frac{z(w\_i)}{0.001}}+1\right) \frac{0.001}{z(w\_i)}$$
![sampling rate](http://i2.muimg.com/567571/fae4be3e93caaa6f.png)

### Negative sampling
Negative sampling addresses the problem (**tremendous number of weight**) by having each training sample only modify a small percentage of the weights, rather than all of them.

When training the network on the word pair (_fox_,_quick_), output neuron corresponding to _quick_ should output 1 (positive), and for all of the **other** output neurons should output 0 (negative). With negative sampling, we are instead going to randomly select just a small number of "negative" words to update the weights for. We will also still update the weights for our "positive" word.

Recall that the output layer of our model has a weight matrix that's $300 \times 10000$. So we will just be updating the weights for our positive word ("quick"), plus the weights for 5 other words that we want to output 0. That's a total of 6 output neurons, and 1800 weight value total. That's only $0.06\%$ of the 3M weights in the output layer.

In the hidden layer, only the weights for the input word are updated.

**Selecting Negative Samples**
The probability for selecting a word as a negative sample is related to its frequency, with more frequent words being more likely to be selected as negative samples.
$$ P(w\_i) = \frac{f(w\_i)^{3/4}}{\sum\_{j=0}^n \left( f(w\_j)^{3/4}\right)}$$

## Summary of word2vec
* Go through each word of the whole corpus
* Predict surrounding words of each word
* This captures co-occurrence of words one at a time

## Evaluation word vectors
* Related to general evaluation in NLP: Intrinsic vs extrinsic
* Intrinsic:
    * Evaluation on a specific/intermediate subtask
    * Fast to compute
    * Helps to understand that system
    * Not clear if really helpful unless correlation to real task is established
* Extrinsic:
    * Evaluation on a real task
    * Can take a long time to compute accuracy
    * Unclear if the subsystem is the problem or its interaction or other subsystems
    * If replacing exactly one subsystem with another improves accuracy
