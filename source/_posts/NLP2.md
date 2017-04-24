---
title: NLP2 Word Vectors
mathjax: true
date: 2017-04-16 10:41:40
categories: NLP
tags: [machine learning, NLP]
---
reference 
[cs224n](http://web.stanford.edu/class/cs224n/syllabus.html)
# lecture 2 Word Vectors
## Word meaning
Definition: **meaning**
* the idea that is represented by a word, phrase, etc.
* the idea that a person wants to express by using words, signs, etc.
* the idea that is expressed in a word of writing
Commonest linguistic way of thinking of meaning
* signifier $\iff$ signified (idea or thing) = denotation

### One-hot vector(meaning in computer)
Common answer: Use a taxonomy like WordNet that has hypernyms relationships and synonym sets
**Problems with this discrete representation**
* Great as a resource but missing nuances, e.g. _synonyms_
    * adept, expert, good, practiced, proficient, skillful
* Missing new words (impossible to keep up to date):
    * wicked, badness, nifty, crack, ace, wizard, genius, ninja
* Subjective
* Requires human labor to create and adapt
* Hard to compute accurate word similarity
* The vast majority of rule-based and statistical NLP work regards words as atomic symbols

We use usually a localist representation ("one-hot") to represent discrete word, but the different word vector $ a^T b = 0$, which means that our query and document vectors are orthogonal. There is no natural notion of similarity in a set of one-hot vectors

"one-hot" vector could deal with similarity separately;
instead we explore a direct approach where vectors encode it

### Distributional similarity based representations
You can get a lot of value by representing a word by means of its neighbors
> You shall know a word by the company it keeps
We will build a dense vector for each word type, chosen so that it is good at predicting other words appearing in its context

### Basic idea of learning neural network word embeddings
Define a model that aims to predict between a center word $w\_t$ and context words in terms of vectors
$$ p(context | w\_t) = \dots $$
which has a loss function, e.g.
$$ J = 1 - p(w\_{-t} | w\_t ) $$
We look at many positions $t$ in a big language corpus
We keep adjusting the vector representations of words to minimize this loss

### Directly learning low-dimensional word vectors
* Learning representations by back-propagating errors (Rumelhart et al., 1986)
* **A neural probabilistic language model** (Bengio et al., 2003)
* NLP (almost) from Scratch (Collobert & Weston, 2008)
* A recent, even simpler and faster model:
word2vec (Mikolov et al. 2013) $\rightarrow$ intro now

## Main idea of word2vec
**Predict between every word and its context words**
Two algorithms
1. **Skip-grams(SG)**
    Predict context words given target (position independent)
1. Continuous Bag of Words(CBOW)
    Predict target from bag-of-words context

Two (moderately efficient) training methods
1. Hierarchical softmax
1. Negative sampling
**Naive softmax**

### The skip-gram model
reference: [Skip-gram tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
Word2vec uses a trick that we train a simple neural network with a single hidden layer to perform a certain task(**Fake Task**), but then we're not actually going to use that neural network for the task we trained it on!
Instead, the goal is actually just to learn the weights of the hidden layer (**Similar to auto-encoder**)

**Fake Task**
_Task goal_ : Given a specific word in the middle of a sentence, look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the "nearby word" that we chose.
> When I say "nearby", there is actually a "window size" parameter to the algorithm. A typical window size might be 5, meaning 5 words behind and 5 words ahead

A sample, window size = 5
![sample](http://i4.buimg.com/567571/4ec276dc62e7c4c8.png)
![another explanation](http://i2.muimg.com/567571/61b440b00b028d9f.png)

**Model detail**
* Input: one-hot vector(dimension means the scale of vocabulary)
* Hidden layer: the word vector for picked word
* Output layer: softmax layer, probability that a randomly selected nearby word is that vocabulary word
![skip gram](http://i1.piimg.com/567571/297e4570b1723090.png)

> For example, we're going to say that we're learning word vectors with 300 features. So the hidden layer is going to be represented by a weight matrix with 10000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron)

![word vector](http://i1.piimg.com/567571/ae499869a4a9ea58.png)
So the end goal of all of this is really just to learn this hidden layer weight matrix.
**one-hot vector $\times$ hidden layer weight matrix $\iff$ lookup table**
![lookup table](http://i2.muimg.com/567571/314a54b706896593.png)

**objective function**
For each word $t=1,\dots,T$, predict surrounding words in a window of "radius" $m$ of every word.

Maximize the probability of any context word given the current center word
$$ J'(\theta) = \prod\_{t=1}^{\pi} \prod\_{-m \le j \le m, j \neq 0 } p \left(w\_{t+j} | w\_t; \theta \right) $$
Negative Log likelihood
$$ J(\theta) = -\frac{1}{T} \sum\_{t=1}^{T} \sum\_{-m \le j \le m, j \neq 0} \log p \left( w\_{t+j} | w\_{t} \right) $$
Where $\theta$ represents all variable we will optimize
