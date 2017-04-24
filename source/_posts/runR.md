---
title: runR
mathjax: true
date: 2016-11-13 15:30:07
categories: software setting
tags: [software,statistical]
---
## Linux下使用R
### 运行linux下的R脚本
#### 编写R文件
1. 新建后缀名为R的文件
2. 写入R程序
3. 在脚本首行加入
```bash
#!/usr/bin/Rscript
```
#### 运行R文件
这里有两种方式

##### 进入R的环境
运行文件
```bash
> source('test.R')
```
注意到在R的环境里面运行脚本，可以保持变量仍然处于环境中

##### 直接在终端中运行
```bash
sudo chmod +x test.R %加入执行权限
./test.R %运行R
```