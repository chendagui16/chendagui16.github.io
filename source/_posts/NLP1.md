---
title: NLP1 Introduction to NLP and DL
mathjax: true
date: 2017-04-16 09:33:24
categories: NLP
tags: [machine learning, NLP]
---
reference
[cs224n](http://web.stanford.edu/class/cs224n/syllabus.html)
# lecture 1 Introduction to NLP and DL
## Natural Language Processing (NLP)
NLP is a field at the intersection of cs, ai and linguistics
**Goal**: for computers to process or "understand" natural language in order to perform tasks that are useful
* Performing Tasks, like making appointments, buying things
* Question Answering

**NLP levels**
![NLP levels](http://i1.piimg.com/567571/71b2c4e1c7c09f52.png)

### Application
* Spell checking, keyword search, finding synonyms
* Extracting information 
* Classifying
* Machine translation
* Spoken dialog systems
* Complex question answering
* ...

### Human language
A human language is a system specifically constructed to convey the speaker/writer's meaning
* No just an environment signal, it's a deliberate communication
* Using an encoding which little kids can quickly learn(**amazingly**)
A human language is a **discrete/symbolic/categorical signaling system**

The categorical symbols of a language can be encoded as a signal for communication in several ways:
* Sound
* Gesture
* Images(writing)
_The symbol is invariant_ across different encodings!

The large vocabulary, symbolic encoding of words creates a problem for machine learning-**sparsity**!

## Deep learning
The first breakthrough results of "deep learning" on large datasets happened in speech recognition
> Context-Dependent Pre-trained Deep Neural Network for Large Vocabulary Speech Recognition, Dahl et al.(2010)

## Why is NLP hard
* Complexity in representing, learning and using linguistic/situational/world/visual knowledge
* Human languages are ambiguous (unlike programming and other formal languages)
* Human language interpretation depends on real world, common sense, and contextual knowledge

## Deep NLP = Deep Learning + NLP
Combine ideas and goals of NLP with using representation learning and deep learning methods to solve them
Several big improvements in recent years in NLP with different
* Levels: speech, words, syntax, semantics
* Tools: parts-of-speech, entities, parsing
* Applications: machine translation, sentiment analysis, dialogue agents, question answering

**Representations of NLP levels: Semantics**
* Traditional: Lambda calculus
    * Carefully engineered functions
    * Take as inputs specific other functions
    * No notion of similarity or fuzziness of language
* DL:
    * Every word and every phrase and every logical expression is a vector
    * A neural network combines two vectors into one vector

**NLP Application: Sentiment Analysis**
* Traditional: Curated sentiment dictionaries combined with bag-of-words representations(ignoring word order) or hand designed negation features
* Same deep learning models that was used for morphology, syntax and logical semantics can be used! $\rightarrow$ RecursiveNN

**Question Answering**
* Traditional: A lot of feature engineering to capture world and other knowledge, e.g., regular expressions, Berant et al.(2014)
* DL: Again, a deep learning architecture can be used!
* Facts are stored in vectors

**Dialogue agents/Response Generation**
* A simple, successful example is the auto-replies available in the Google Inbox app
* An application of the powerful, general technique of _Neural Language Models_, which are an instance of RNN

**Machine Translation**
* Many levels of translation have been tried in the past
* Traditional MT systems are large complex systems
