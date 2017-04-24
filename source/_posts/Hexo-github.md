---
title: Hexo_github
date: 2016-09-05 16:02:22
categories: software setting
tags: 
	- hexo
	- githup pages
	- blog
---
## Hexo and github pages SETTING
### Introduction
----
因为在调试的过程中，经常会发现一些bug，而这些bug过一段时间就会遗忘，为了记住这些bug，因此开始学会养成写个人技术blog的习惯。
　　
这里采用的是hexo 来解析markdown文件，然后将其发布到github pages上。然而在设置的时候有出现了一些问题，这里记录这些问题。

### Setting
----
#### 安装[hexo](https://hexo.io/docs/index.html)
##### 安装Node.js
先安装[nvm](https://github.com/creationix/nvm)
```bash
$ curl https://raw.github.com/creationix/nvm/master/install.sh | sh
```
nvm 安装完之后，**重启终端**，然后安装Node.js
```bash
$ nvm install stable
```
也可以通过[源码](http://nodejs.org/)来安装
##### 安装git
```bash
$ sudo apt-get install git-core
```
##### 安装hexo
```bash
$ npm install -g hexo-cli
```
##### 安装hexo必备的服务
安装server和git服务
```bash
$ cd (blog workspace)
$ npm install hexo-server　 --save
$ npm install hexo-deployer-git --save
```
#### 设置github pages
github账号下，需要建立对应的仓库，其名字必须为(username).github.io。否则将无法得到正确的github pages显示。
#### 设置hexo
初始化blog文件夹
```bash
$ cd (blog workspace)
$ hexo init
```
编辑配置文件*_config.yml*
按照描述更改配置，注意要更改deploy选项。
> deploy:
         　type: git
         　repository: git@github.com:chendagui16/chendagui16.github.io.git	  
         　branch: master
 
 这里我使用了ssh协议        　
#### 更换主题与数学公式
我采用了[jacman](https://github.com/wuchong/jacman)的主题，设置方法可进入github中观看。
>　这个主题中提供了较多的选项，包括如何解决数学公式的显示问题，然而，我只得到了本地页面的显示并没有得到github pages上的显示，问题仍未解决。谁知道了能否告诉我怎么解决。

在jacman的主题下，设置数学显示包括两步，第一步是更改**主题**内的*_config.yml*文件。
```yml
mathjax: true
```
第二步是在写front-matter内添加一行
```yml
mathjax: true
```
### 命令
---
```bash
$ hexo init 初始化
$ hexo new [layout] <title>　新建文档，layout提供默认版式
$ hexo g[enerate]　[-d]　渲染成静态网页
$ hexo s[erver] 打开本地服务器
$ hexo d[eploy] 推送到github
$ hexo clean　清除原静态文件
```