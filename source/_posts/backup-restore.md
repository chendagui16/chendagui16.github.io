---
title: backup-restore
mathjax: true
date: 2016-09-10 19:25:29
categories: work tips
tags: [ubuntu,backup]
---

## ubuntu14.04备份与恢复系统
---
### 备份
首先清空回收站的垃圾，然后使用终端清理一些缓存
```bash
sudo apt-get autoclean
sudo apt-get clean
sudo apt-get autoremove
```
然后新开一个终端
```bash
sudo su #su模式
cd /  #进入根文件夹
tar cvpzf backup.tgz --exclude=/proc --exclude=/lost+found --exclude=/backup.tgz --exclude=/mnt --exclude=/sys --exclude=/home/dagui/Downloads / 
```
注意到，因为我的文件夹中Downloads包含大量的下载文件，而我不想将它包含在备份的文件夹中，因此我排除了它。
> 命令说明：
c – 创建一个新的备份文件
v – 详细模式，将执行过程全部输出到屏幕
p – 保留文件的权限信息以便恢复
z – 使用gzip压缩文件，以便减小体积
f <filename> – 指定备份文件的名称

### 恢复
在一个可用的系统中
```bash
sudo su
```
拷贝backup.tgz到根目录下,然后执行命令
```bash
tar xvpfz backup.tgz -C / 
```
然后重新创建之前剔除的目录
```bash
mkdir /proc /lost+found /mnt /sys /home/dagui/Downloads
```

### 参考链接
[Ubuntu 14.04如何备份和恢复系统](http://www.lh126.net/article-view-235.html)
[Ubuntu Server服务器备份与还原备份命令](http://www.111cn.net/sys/202/51307.htm)