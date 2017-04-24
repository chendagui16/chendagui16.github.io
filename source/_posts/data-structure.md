---
title: data structure
mathjax: true
date: 2017-04-14 13:50:09
categories: algorithm
tags: [algorithm, data structure]
---
# data structure
## 二叉搜索树
二叉搜索树是一个满足以下性质的二叉树
> 对于任何结点$x$, 其左子树中的关键字最大不超过$x.key$, 其右子树中的关键字最小不低于$x.key$. 不同的二叉搜索树可以代表同一组值的集合, 大部分搜索树的最坏运行时间与树的高度成正比

### 遍历
先序遍历(preorder tree walk): 根-左-右
中序遍历(inorder tree walk): 左-根-右
后序遍历(postorder tree walk): 左-右-根

根据二叉搜索树的性质, 通过中序遍历可以按序输出所有关键字
```
INORDER-TREE-WALK(x):
if x!= NIL
    INORDER-TREE-WALK(x.left)
    print x.key
    INORDER-TREE-WALK(x.right)
```
中序遍历需要$\Theta(n)$的时间

### 查询
```
TREE-SEARCH(x, k):
if x==NIL or k==x.key
    return x
if k<x.key
    return TREE-SEARCH(x.left, k)
else
    return TREE-SEARCH(x.right, k)
```
使用循环代替递归来重写这个程序
```
ITERATIVE-TREE-SEARCH(x, k):
while x != NIL and k != key
    if k < x.key
        x = x.left
    else
        x = x.right
return
```

**最大关键字元素和最小关键字元素**
```
TREE-MINIMUM(x):
while x.left != NIL
    x = x.left
return x
```

```
TREE-MAXIMUM(x):
while x.right != NIL
    x = x.right
return x
```

**后继和前驱**
```
TREE-SUCCESSOR(x):
if x.right != NIL
    return TREE-MINIMUM(x.right)
y = x.p
while y != NIL and x == y.right
    x = y
    y = y.p
return y
```

**在一颗高度为$h$的二叉搜索树上, 动态集合上的操作SEARCH, MINIMUM, MAXIMUM, SUCCESSOR, PREDECESSOR可以在$O(h)$时间内完成**

### 插入和删除
**Insert**
把一个新值$v$插入到一颗完全二叉搜索树中, 需要调用过程TREE-INSERT. 该过程以节点$z$作为输入, 其中$z.key=v, z.left=NIL, z.right=NIL$, 这个过程需要修改$T,z$的某些属性, 来把$z$插入到树的相应位置上
```
TREE-INSERT(T,z):
y = NIL
x = T.root
while x != NIL
    y = x
    if z.key < x.key
        x = x.left
    else
        x = x.right
z.p = y
if y == NIL
    T.root = z
elif z.key < y.key
    y.left = z
else
    y.right = z
```
**Delete**
从一棵二叉搜索树$T$中删除一个结点$z$的整个策略分为四种情况
* 如果$z$没有左孩子, 那么用右孩子来替换$z$. 这里不管右孩子是NIL还是具体的结点
* 如果$z$仅有左孩子, 那么用左孩子来替换$z$
* 如果$z$有两个孩子, 我们要查找$z$的后继$y$, 这个后继位于$z$的右子树中并且没有左孩子, 现在需要将$y$移出原来的位置进行拼接, 并替换树中的$z$
* 如果$y$是$z$的右孩子, 那么用$y$替换$z$, 并仅留下$y$的右孩子

为了在二叉搜索树内移动子树, 定义一个子过程TRANSPLANT, 它是用另一棵子树替换一棵子树并成为其双亲的孩子结点. 当TRANSPLANT用一棵以$v$为根的子树来替换一棵以$u$为根的子树时, 结点$u$的双亲就变成了结点$v$的双亲, 并且最后$v$成为$u$的双亲的相应孩子.

```
TRANSPLANT(T,u,v):
if u.p == NIL
    T.root = v
elif u == u.p.left
    u.p.left = v
else 
    u.p.right = v
if v != NIL
    v.p = u.p
```
利用TRANSPLANT过程,我们建立从二叉树$T$中删除结点$z$的算法
```
TREE-DELETE(T,z)
if z.left == NIL
    TRANSPLANT(T, z, z.right)
elseif z.right == NIL
    TRANSPLANT(T, z, z.left)
else y = TREE-MINIMUM(z.right)
    if y.p != z
        TRANSPLANT(T, y, y.right)
        y.right = z.right
        y.right.p = y
    TRANSPLATNT(T, z, y)
    y.left = z.left
    y.left.p = y
```

**在一棵高度为$h$的二叉搜索树上, 实现动态集合操作INSERT和DELETE的运行时间均为$O(h)$**

## 图
### 图的表示
对于图$G=(V,E)$, 可以用两种标准表示方法表示: **邻接链表和邻接矩阵**

**邻接链表**
适合在稀疏图(边的条数$|E|$远远小于$|V|^2$的图), 对于图$G=(V,E)$来说, 其邻接链表表示由一个包含$|V|$条链表的数组$Adj$所构成, 每个结点有一条链表. 对于每个结点$u \in V$, 邻接链表$Adj[u]$包含所有与结点$u$之间有边连接的结点$v$, 即$Adj[u]$包含图$G$中所有与$u$邻接的结点
![邻接链表](http://i1.piimg.com/567571/0b5e457f3d662cbe.jpg)
如果$G$是一个有向图, 则所有邻接链表的长度之和等于$|E|$, 如果$G$是一个无向图, 所有邻接链表的长度之和等于$2|E|$

邻接链表稍加修改即可以表示权重图, 我们可以直接将边的权重值存放在结点的邻接链表里. 邻接链表表示法的鲁棒性很高, 可以对其进行简单的修改来支持许多其他的图变种

**邻接矩阵**
把图$G$中的结点编号为$1,2,\dots,|V|$, 则邻接矩阵可以有一个$|V| \times |V|$来表示, 该矩阵满足以下条件
$$ a_{ij} = \begin{cases} 1 & \text{若} (i,j) \in E \\\\ 0 & \text{其他} \end{cases} $$
无向图的邻接矩阵是对称的

### 广度优先搜索
给定图$G=(V,E)$和一个可以识别的源结点$s$, 广度优先搜索对图$G$中的边进行系统性的探索来发现可以从源节点$s$到达的所有结点. 该算法始终是将已发现结点和未发现结点之间的边界, 沿其广度方向向外拓展, 算法需要在发现所有距离源结点$s$为$k$的所有结点之后, 才会发现距离源结点$s$为$k+1$的其他节点.

广度优先搜索会给结点染色(白色, 黑色, 灰色), 白色表示未发现的结点, 灰色和黑色的结点表示已被发现的结点, 灰色表示该结点周围存在着未被发现的白色结点, 黑色表示该结点周围的结点都已经被发现了. 在执行广度优先搜索的过程中将构造出一棵广度优先树. 如果我们通过结点$u$第一次搜索到了$v$, 那么称$u$是$v$的前驱或父结点.

下面给出广度优先搜索的算法, 其中我们把每个结点$u$的颜色存在属性$u.color$里, 将$u$的前驱结点存放在属性$u.pi$里, 如果$u$没有前驱结点(例如, $u=s$或者尚未被发现), 则$u.\pi = NIL$. 属性$u.d$记录的是广度优先搜索算法所计算出的从源结点$s$到结点$u$之间的距离. 该算法使用一个先进先出的队列$Q$来管理灰色结点集.
```
BFS(G, s):
for each vertex u in G.V - {s} 
    u.color = WHITE
    u.d = inf
    u.pi = NIL
s.color = GRAY
s.d = 0
s.pi = NIL
Q = empty
ENQUEUE(Q, s)
while Q != empty
    u = DEQUEUE(Q)
    for each v in G.Adj[u]
        if v.color == WHITE
            v.color = GRAY
            v.d = u.d + 1
            v.pi = u
            ENQUEUE(Q, v)
    u.color = BLACK
```

### 深度优先搜索
深度优先搜索只要可能就在图中尽量深入. 深度优先搜索总是对最近才发现的结点$v$的出发边进行探索, 知道该结点的所有出发边都被发现为止. 一旦结点$v$的所有出发边都被发现, 搜索则_回溯_到$v$的前驱结点, 来搜索该前驱结点的出发边. 像广度优先搜索一样, 在对已被发现的结点$u$的邻接链表进行扫描时, 每当发现一个结点$v$时, 深度优先搜索算法将对这个事件进行记录, 将$v$的前驱属性$v.\pi$设置为$u$. 与广度优先搜索不同的是, 广度优先搜索的前驱子图形成一棵树, 而深度优先搜索的前驱子图可能由多棵树组成, 因为搜索可能从多个源结点重复进行. 深度优先搜索的前驱子图形成一个由多颗深度优先树构成的深度优先森林.

除了创建一个深度优先森林外, DFS还在每个结点上盖一个时间戳. 每个结点有两个时间戳: $v.d$记录结点$v$第一次被发现的时间(涂上灰色的时候), $v.f$记录完成对$v$的邻接链表扫描的时间(涂上黑色的时候). 显然结点$u$在时刻$u.d$之前为白色, 在时刻$u.d$和$u.f$之间为灰色, 在时刻$u.f$之后为黑色.

```
DFS(G):
for each vertex u in G.V
    u.color = WHITE
    u.pi = NIL
time = 0
for each vertex u in G.V
    if u.color == WHITE
        DFS-VISIT(G, u)

DFS-VISIT(G, u):
time = time + 1
u.d = time
u.color = GRAY
for each v in G.Adj[u]
    if v.color == WHITE
        v.pi = u
        DFS-VISIT(G, v)
u.color = BLACK
time = time + 1
u.f = time
```
