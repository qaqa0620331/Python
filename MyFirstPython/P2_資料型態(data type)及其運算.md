# Python程式設計{基礎課程}

# 2_1.各種資料型態(data type)及其運算[1][2][3][4]...[X]
```
識別字與關鍵字

數字型(numeric)資料型態及其運算
Integral 類型:整數類型|布爾型
浮點類型:浮點數|複數
二進位八進位十進位十六進位數字

字串(string)資料型態及其運算

文數字編碼與解碼
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

# 2.2.數字型(numeric)資料型態及其運算

#### 餘數運算子 (remainder|modulo)
```
題目:將以秒表示的總時間長，分別取得分鐘數及剩餘的秒數。
```
```
#!/usr/bin/env python
#coding=utf-8

seconds = eval(input("Enter an integer for seconds: "))

minutes = seconds // 60     # Find minutes in seconds
remainingSeconds = seconds % 60   # Seconds remaining
print(seconds, "seconds is", minutes, "minutes and", remainingSeconds, "seconds")
```

# 2.3:二進位,八進位,十進位與十六進位

>* Python有許多內建函數(built-in functions)
>* https://docs.python.org/3/library/functions.html
>* https://docs.python.org/2/library/functions.html
>* https://www.programiz.com/python-programming/methods/built-in

```
題目:將十六進位的0xabcd寫出其二進位,八進位,十進位的表示
```

# 2.4. 字串(string)資料型態及其運算

```
>* python提供內建的str字串類別
>* 字串物件是不可變更的。一旦被建立，其內容就不可以改變。
>* 為了效率最佳化，若字串內容相同，則Python只使用一物件。
```

#### 建立字串(string)==>使用引號'或"來創建字串
```
string = ""
stringa = str()
id(string)
id(stringa)

stringb = str("MyfirstCTF")

string1 = "I love you"
```

#### str 類別有許多方法(運算子)與函數
```
len(stringb)
max(stringb)
min(stringb)
索引運算子[]
```

#### str 類別的+、 *、 [ : ] 與 in 運算子
```
s = '}FTC NOCTIH ot emocleW{noctih'

''.join(reversed(s))
```
#### replace()取代
```
string1 = "I love you"
string1.replace("love","hate")
```
#### join()
```
'_'.join("Dragon")
'x'.join("Dragon")
```
#### split()
```
string1 = "I love you"
string1.split()

#string2 = 'x'.join("Dragon")
#string2.split(sep='x')
```
#### upper() swapcase()
```
string1.upper()
string1.swapcase()
```
### 有空白==>去除空白
```
string11 = "   I love you  "
string11.strip()
```
### 索引(indexing)與切片(slicing)
```
s = 'Hello World'
s[0]
s[4]
s[-2]
s[1:3]
s[1:-2]
s[0:3]
s[::2]
s[::-1]
```


### 字串(string)的應用:編碼與解碼

>* 美國 ASCII編碼
```
僅對 10 個數字、26 個大寫英文字母、26個小寫英文字母， 以及一些符號進行編碼。 
ASCII 採用 1 個byte編碼字元，最多只能表示 256 個符號。
```

>* 各國文字的編碼:UTF-8 、UTF-16 、UTF-32 、GB2312 、GBK 、CP936 、Big5 、base64 
>* UTF-8 是國際適用的編碼， 以l個位元組表示英語字元(相容於 ASCII)， 以 3個byte(位元組)表示中文
>* Python 3.x 完全支援中文，並使用 Unicode 編碼格式，無論是數字、英文字母，還 是中文字等，都以一個字元對待和處理。 
>* Python 3.x 甚至可以使用中文作為變數名稱 

### 如何操縱Unicode字串[Python3]
```
s=u'台南'
type(s)
len(s)
s


s1='台南'
len(s1)

s2='Dragon'
len(s2)

s3='台南Dragon'
len(s3)

# 中文與英文字元同等對待，都算一個字元

# 可以使用中文作為變數名稱

地點='台南'
學校='台南一中'
年代=2018

print(地點)
print(學校)
print(年代)
```

#### Python程式的字串編碼請以#coding 明確指定
```
#coding  = utf-8
#coding:utf-8
#-*-coding:utf-8 -*-
```
#### Python 支援短字串駐留機制(不支援長字串駐留機制)
```
a='20180605'
b='20180605'
id(a) == id(b)


a2='20180605'*100
b2='20180605'*100
id(a2) == id(b2)
```
#### 使用內建方法 isinstance()或 type()來判斷一變數是否為字串
```
type('台南')

type('台南'.encode('Big5'))

isinstance('台南',str)

type('台南') == str

```

#### 跳拖字元

在字串某些特定的符號前加一個斜線之後，該字元將被解釋為另外一種涵義，不再代表本來的字元。

>* \b  ==>
>* \f  ==>
>* \n  ==>斷行
>* \t  ==>

```
print('Hello\nWorld')


```

```
ord('龍')
hex(_)
print('\u    ')
```
四個十六進位表示的 Unicode 字元


#### 字串(string)格式化:兩種方法

方法1:使用%

```
zzz=2018

s_2 = "%b"%zzz #二進位
s_8 = "%o"%zzz
s_10 = "%i"%zzz
s_16 = "%x"%zzz
```
```
'%d,%c'%(68,68)
```


方法2:使用format()

更靈活，不僅能透過位置， 還支援與位置無關的參數名稱進行格式化， 以及序列拆解格式化字串等，為程式人員提供極大的便利性

```
2/3

print('{0:.3f}'.format(2/3))

```
```
print ("The number {0:,} in hex is:{0:#x}, in oct is:{0:#o}".format (5566))
```
```
print ("The number {0:,} in hex is:{0:#x},The number {1} in oct is:{1:#o}".format (5566,56))
```
```
print ("The number {1} in hex is:{1:#x},The number {0} in oct is:{0:,:#o}".format (5566,56))
```

### 字串(string)的應用:破密分析

擴充版凱薩加密及暴力破解
```
#!/usr/bin/env python3

alpha = 'abcdefghijklmnopqrstuvwxyz'
num = '0123456789'
alnum = alpha + num

ctext = '7sj-ighm-742q3w4t'

def rotate(s, num):
    new1 = ''
    for c in s:
        if c in alnum:
            new1 += alnum[(alnum.index(c) + num) % 36]
        else:
            new1 += c
    return new1

for x in range(36):
    print("{}".format(rotate(ctext, x)))

```

```
#!/usr/bin/env python3
import string

alphabet =  string.ascii_lowercase + string.ascii_uppercase + string.digits 

ctext = "7sj-ighm-742q3w4t"

def shift(n):
    message = ""
    for index, char in enumerate(ctext):
        if char == "-":
            message += char
        else:
            message += alphabet[(alphabet.index(ctext[index])+n)%len(alphabet)]
    return message.upper()

for i in range(len(alphabet)):
    message = shift(i)
    if "RC3" in message:
        print(message)
```
```
a='cvqAeqacLtqazEigwiXobxrCrtuiTzahfFreqc{bnjrKwgk83kgd43j85ePgb_e_rwqr7fvbmHjklo3tews_hmkogooyf0vbnk0ii87Drfgh_n kiwutfb0ghk9ro987k5tfb_hjiouo087ptfcv}'

a=a[3:]
flag = ''
for x in range(0,len(a),1):
    if x%5==0:
        flag+=a[x]
print flag
```

re模組與正規表達法(NeXT)


# 2_5.列表|串列(list)資料型態及其運算

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
list5 = list3[2:4]
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
### list的超強大功能:序列拆解
```
a=[12,212,321,325]
x,y,z,k = a
z
```

### list的超強大功能list comprehension列表生成式 
```
A list comprehension consists of brackets containing an expression followed by a for clause, 
then zero or more for or if clauses. 
The result will be a new list resulting from evaluating the expression 
in the context of the for and if clauses which follow it. 

https://en.wikipedia.org/wiki/List_comprehension
```

```
list1 = [x for x range(0, 5)] # Returns a list of 0, 1, 2, 4
list1 
list2 = [0.5 * x for x in list1] 
list2
list3 = [x for x in list2 if x < 1.5]
list3
```

# 綜合練習
```
import random
import time

x = list(range(10000))		#產生成列表
y = set(range(10000))		#產生成集合
z = dict(zip(range(1000),range(10000)))		#產生成字典
r = random.randint(0, 9999)	#產生成亂數

start = time.time()
for i in range(9999999):
    r in x 				#測試清單列表中是否包含某個元素
print('list,time used:', time.time()-start)

start = time.time()
for i in range(9999999):
    r in y					#測試集合中是否包含某個元素
print('set,time used:', time.time()-start)

start = time.time()
for i in range(9999999):
    r in z					#測試字典中是否包含某個元素
print('dict,time used:', time.time()-start)
```
