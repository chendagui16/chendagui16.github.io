---
title: ubuntu-bugs
mathjax: true
date: 2016-09-23 20:59:36
categories: work tips
tags: [ubuntu,bugs]
---
## ubuntu中的error与solutions
这里记录一些在ubuntu使用过程中碰到的errors，并记录相应的解决办法。
### CPG ERROR
在/etc/apt/source.list中添加一些源时，执行
```bash
sudo apt-get update
```
若没有这个源对应的秘钥时，会产生error，其中error信息如下：
> GPG error:  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 06F90DE5381BA480

这里我们可以添加对应的秘钥，即可解决问题。
```bash
gpg --keyserver keyserver.ubuntu.com --recv 06F90DE5381BA480
gpg --export --armor 06F90DE5381BA480  | sudo apt-key add -
```