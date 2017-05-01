---
title: pickle
mathjax: true
date: 2017-05-01 22:14:51
categories: work tips
tags: [python]
---

 # pickle 模块的简易使用

## 序列化存储

```python
import pickle
data = ...
f = open('file.pkl','wb')
pickle.dump(data, f)
```

## 序列化读取

```python
import pickle
pkl_file = open('file.pkl','rb')
data = pickle.load(pkl_file)
```

