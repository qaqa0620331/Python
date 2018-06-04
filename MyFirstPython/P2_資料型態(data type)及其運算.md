# Python程式設計{基礎課程}

# [2]各種資料型態(data type)及其運算[1][2][3][4]...[X]
```
識別字與關鍵字

數字型(numeric)資料型態及其運算
Integral 類型:整數類型|布爾型
浮點類型:浮點數|複數
二進位八進位十進位十六進位數字

字串(string)資料型態及其運算
如何操縱Unicode字串

Collection Data Types組合類型資料型態 
列表|串列(list)資料型態及其運算
元組(tuple)資料型態及其運算
辭典|字典(dic)資料型態及其運算
集合(set)資料型態及其運算
```
### 運算子:
>* Arithmetic Operators算術運算子
>* 餘數運算子 (remainder|modulo)
>* Membership Operator
>* Comparison Operators
>* Logical Operators邏輯運算子
>* 關係運算子

### 組合類型資料型態的高級特性:切片|迭代 |列表生成式 |生成器|迭代器

# 列表|串列(list)資料型態及其運算

>* python提供內建的list類別

### 建立list
```
list1 = list() 
list1 = [] 

list2 = list([2, 3, 4])
list2 = [2, 3, 4] 

list3 = list(["red", "green", "blue"]) 
list4 = list(range(3, 6))
list5 = list("abcd") 
```

### list的各種函數1
```
[動手做]下列範例會產生何種結果
list1 = [21, 33, 14, 12, 32,98]

len(list1)
max(list1)
min(list1)
sum(list1) 
```

### list的各種函數2:串列的加法|乘法與分割運算

[動手做]下列範例會產生何種結果
```
list1 = [21, 23, 25]
list2 = [11, 92]

list3 = list1 + list2
list3
list4 = 3 * list1
list4
list5 = list3[2 : 4]
list5
```
### list的各種函數3:串列的分割運算
```
items = "Welcome to the MyFirstCTF".split() 
print(items)

items = "314#123#738#445".split("#")
print(items)
```
### list的各種函數4:more
```
加一元素到串列的尾端:append()
回傳元素x 出現於串列的次數:count()
附加在l中所有元素於串列:extend()
回傳在串列中第一次出現元素x的索引:index()
將串列中的元素加以反轉:reverse()
由小至大排序串列中的元素:sort()
```

### list的超強大功能list comprehension列表生成式 
A list comprehension consists of brackets containing an expression followed by a for clause, 
then zero or more for or if clauses. 
The result will be a new list resulting from evaluating the expression 
in the context of the for and if clauses which follow it. 

https://en.wikipedia.org/wiki/List_comprehension

list1 = [x for x range(0, 5)] # Returns a list of 0, 1, 2, 4
list1 
list2 = [0.5 * x for x in list1] 
list2
list3 = [x for x in list2 if x < 1.5]
list3
------------------------------------------------
