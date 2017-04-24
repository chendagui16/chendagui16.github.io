---
title: texstudio-bugs
mathjax: true
date: 2016-09-26 18:49:31
categories: Latex
tags: [Latex,bugs]
---
## Latex使用技巧
记录一些使用latex的过程中出现的问题。
> 使用的环境是linux英文系统下的texstudio。

### texstudio中的中文问题
首先要更新ubuntu的对中文的语言支持，确保存在着中文的字体，同时安装中文输入法。
然后针对texstudio，设置分为三步。
##### 1.改变build命令，将编译器改成XeLaTeX
##### 2.将编辑器的默认编码改成UTF-8
##### 3.修改tex文件

```tex
\documentclass[UTF8,12pt]{report} %加入UTF-8设置
\usepackage{ctex}%使用ctex宏包
```
> 因为我主要使用英文的系统，所以这里只提供了基本的中文支持。详细的中文使用得需要参考google。