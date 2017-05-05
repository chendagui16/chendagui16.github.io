---
title: OCR
mathjax: true
date: 2017-05-05 19:19:33
categories: OCR
tags: [deep learning, OCR]
---

# OCR 文字光学识别 

## 问题描述

* 文字检测
* 文字识别

## 文字检测(localization)

### 特点

* IOU不是一个好的criterion, 检测到一部分文字也行
* various fonts, colors, languages and bks etc.
* perspective transformation
* layouts
* word/line level

### methods

* tranditional method
* deep learning
  * RPN based, detection
  * FCN based, segmentation 

Note: Sence text detection via holistic, multi-channel prediction

## 文字识别

### method

* CNN/MDLSTM + RNN + CTC
* Sequence to Sequence with Attention
* Combine CTC and Attention

note: CTC用来将文字进行对齐

note: LSTM -> GRU ->EURNN

#### RNN 

* Bidirectional RNN (文字识别中经常使用)
* Stack RNN(百度 7个堆叠， 谷歌5个堆叠)
* MDLSTM/Grid LSTM

#### challenge

* Chinese include too many characters
  * Uncoutable labels
  * Insufficient data (Synthesize)
  * Much more computation
* Incaptable
  * Too much perspective transform (STN)
  * Vertical layout