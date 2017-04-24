---
title: sortAlgorithm
mathjax: true
date: 2017-04-11 09:15:32
categories: algorithm
tags: [algorithm]
---
# Several pseudo-code for sort algorithm 
## bubble sort
worst time complexity: $O(n^2)$
best time complexity $O(n)$
average time complexity $O(n^2)$
additional space complexity $O(1)$
```
bubble_sort(list, p, r):
for i = p to r-1
   for j = i+1 to r
     if list[i] > list[j]
         swap(list[i],list[j])
```

## selection sort
average time complexity $O(n^2)$
additional space complexity $O(1)$
```
selection_sort(list, p, r):
for i = p to r-1
    min = i
    for j = i+1 to r
        if list[j] < min
            min = j
    if min!=i
        swap(list[i], list[min])
```

## Merge sort
time complexity: $O(n\log n)$
space complexity: $O(n)$
```
merge_sort(list, p, r):
if p >= r
    return
mid = ceil(p+r)
merge_sort(list, p, mid)
merge_sort(list, mid+1, r)
i = p
j = mid+1
k = 1
while(i<=mid && j <= r)
    if list[i] > list[j]
        tmp[k++] = list[j++]
    else
        tmp[k++] = list[i++]
while(i<=mid)
    tmp[k++] = list[i++]
while(j<=r)
    tmp[k++] = list[j++]
list[p..r] = tmp
```

## Quick sort
worst time complexity: $O(n^2)$
best time complexity: $O(n\log n)$
average time complexity: $O(n \log n)$
additional space complexity: $O(1)$
```
quick_sort(list, p, r):
if p>=r
    return
else
    mid = partition(list, p, r)
    quick_sort(list, p, mid-1)
    quick_sort(list, mid+1, r)

partition(list, p, r):
tmp = list[r]
i = p
j = r-1
while(i<j)
    while(i<j && list[i] <= tmp) 
        i++
    while(i<j && list[j] >= tmp)
        j--
    if(i<j) 
        swap(list[i], list[j])
swap(list[i], list[r])
return i
```

## heap sort
time complexity: $O(n\log n)$
additional space complexity $O(1)$
```
MAX-HEAPIFY(A, i):
l = LEFT(i)
r = RIGHT(i)
if l <= A.heap-size and A[l]>A[i]
    largest = l
else
    largest = i
if r <= A.heap-size and A[r]>A[largest]
    largest = r
if largest != i
    swap(A[i], A[largest])
    MAX-HEAPIFY(A, i)

BUILD-MAX-HEAP(A):
A.heap-size = A.length
for i = ceil(A.length/2) downto 1
    MAX-HEAPFIFY(A,i)

HEAPSORT(A):
BUILD-MAX-HEAP(A)
for i=A.length downto 2
    swap(A[1], A[i])
    A.heap-size = A.heap-size - 1 
    MAX-HEAPIFY(A,1)
```
