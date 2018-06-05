# [4]函式/函數/function

```
函數是組織好的，可重複使用的，用來實現單一，或相關聯功能的程式碼片段。
函數能提高應用的模組性，和代碼的重複利用率。
Python提供了許多內建函數，比如print()。
使用者也可以自己創建函數，這被叫做使用者自訂函數。

定義一個函數
你可以定義一個由自己想要功能的函數，以下是簡單的規則：
函數代碼塊以 def 關鍵字開頭，後接函數識別字名稱和圓括號()。
任何傳入參數和引數必須放在圓括號中間。圓括號之間可以用於定義參數。
函數的第一行語句可以選擇性地使用文檔字串—用於存放函數說明。
函數內容以冒號起始，並且縮進。
return [運算式] 結束函數，選擇性地返回一個值給調用方。不帶運算式的return相當於返回 None。
```

定義語法:函式包含標頭和主體。標頭(header)起源於def關鍵字，後接函式名稱和參數，最後以冒號結尾。 

```
def functionname(parameters):
   "函數_文檔字串"
   function_suite
   return [expression]
```

# 功能==>函數

請分別計算 1 到 10，20 到 37，以及 35 到 49 間整數的總和?

你會怎麼做

方法一:
```
sum = 0
for i in range(1, 10):
    sum += i
print("Sum from 1 to 10 is", sum)

sum = 0
for i in range(20, 37):
    sum += i
print("Sum from 20 to 37 is", sum)
sum = 0
for i in range(35, 49):
    sum += i
print("Sum from 35 to 49 is", sum)
```

方法二:
```
def sum(i1, i2):
    result = 0
    for i in range(i1, i2):
        result += i
    return result

def main():
    print("Sum from 1 to 10 is", sum(1, 10)) 
    print("Sum from 20 to 37 is", sum(20, 37))
    print("Sum from 35 to 49 is", sum(35, 49))

main() # Call the main function
```
形式參數 vs 實質參數(引數)

>* 形式參數==定義於函數標頭的變數
>* 實質參數(引數)==當函式被呼叫時，你必須給予的參數則被稱為
>* 函式會使用關鍵字return 回傳一個結果數值，此數值被稱為回傳值。

```
================================================================
【Python 練習實例47】練習撰寫函式swap(a,b)
寫一個函式將輸入的兩個變數值互換
================================================================
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
def swap(a,b):
    a,b = b,a
    return (a,b)
 
if __name__ == '__main__':
    x = 10
    y = 20
    print 'x = %d,y = %d' % (x,y)
    x,y = swap(x,y)
    print 'x = %d,y = %d' % (x,y)
```

```
#!/usr/bin/python 
# -*- coding: UTF-8 -*- 

def crypt(source, key):
    from itertools import cycle
    result = ''
    temp = cycle(key)
    for ch in source:
        result = result + chr(ord(ch) ^ ord(next(temp)))
    return result

source = 'BreakALLCTF_IknowhowtoXORRRRing'
key = 'HappyHackingHigh'

print('未加密的明文:'+source)
encrypted = crypt(source, key)
print('加密過的密文:'+encrypted)
decrypted = crypt(encrypted, key)
print('解密過的答案:'+decrypted)
print('使用的金鑰:'+ key)
```
# 有無回傳值的函式

無回傳值的函式
```
# Print grade for the score 
def printGrade(score):
    if score >= 90.0:
        print('A')
    elif score >= 80.0:
        print('B')
    elif score >= 70.0:
        print('C')
    elif score >= 60.0:
        print('D')
    else:
        print('F')

def main():
    score = eval(input("Enter a score: "))
    print("The grade is ", end = "")
    printGrade(score)

main() # Call the main function
```

有回傳值的函式
```
# Return the grade for the score 
def getGrade(score):
    if score >= 90.0:
        return 'A'
    elif score >= 80.0:
        return 'B'
    elif score >= 70.0:
        return 'C'
    elif score >= 60.0:
        return 'D'
    else:
        return 'F'

def main():
    score = eval(input("Enter a score: "))
    print("The grade is", getGrade(score))

main() # Call the main function
```
# 匿名函數==>

# 遞迴函式遞迴函式==recursive vs iterative(loop)
```
[程式開發作業]費氏數列
[程式開發作業]n!
[程式開發作業]河內塔
```
```
[程式開發作業]
費氏數列Fibonacci sequence:recursive vs iterative(loop)
又稱黃金分割數列
0、1、1、2、3、5、8、13、21、34、……。
在數學上，費波那契數列是以遞迴的方法來定義：
F(0) = 0     (n=0)
F(1) = 1    (n=1)
F(n) = F(n-1)+ F(n-2)(n=>2)
```

方法一:使用iterative
```
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
def fib(n):
    a,b = 1,1
    for i in range(n-1):
        a,b = b,a+b
    return a
 
print fib(10)
```

方法二:使用recursive
```
#!/usr/bin/python
# -*- coding: UTF-8 -*-

def fib(n):
    if n==1 or n==2:
        return 1
    return fib(n-1)+fib(n-2)

print fib(10)
```
```
================================================================
【Python 練習實例28】
有5個人坐在一起，
問第五個人多少歲？他說比第4個人大2歲。
問第4個人歲數，他說比第3個人大2歲。
問第三個人，又說比第2人大兩歲。
問第2個人，說比第一個人大兩歲。
最後問第一個人，他說是10歲。
請問第五個人多大？
================================================================
```

# Python內建函數==>python內建函式Built-in Functions:

>* https://docs.python.org/2/library/functions.html
```
dir(__builtins__)
help('math')
help(id)
```
```
import sys
print(system.builtin_module_names)
```

Ascii code==> https://zh.wikipedia.org/wiki/ASCII
```
ord('a')
chr(97)
chr(ord('A')+1)
```

# [4.5]使用函式來模組化程式

函式可用來減少多餘的程式碼，允許程式碼重複使用，還可以用來模組化程式碼，提升程式品質

步驟一:先將常用的功能寫成函式MyGCD.py
```
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Return the gcd of two integers 
def gcd(n1, n2):
    gcd = 1 # Initial gcd is 1
    k = 2   # Possible gcd

    while k <= n1 and k <= n2:
        if n1 % k == 0 and n2 % k == 0:
            gcd = k # Update gcd
        k += 1

    return gcd # Return gcd
```

步驟二:在主程式呼叫函式MyGCD.py
```
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from MyGCD import gcd # 載入要用的模組

# 請使用者輸入兩個整數
n1 = eval(input("請輸入第一個整數: "))
n2 = eval(input("請輸入第二個整數 "))

# 輸出兩個整數的最大公因數
print("兩個整數的最大公因數", n1,
    "and", n2, "is", gcd(n1, n2))
```

# 函數開發的更多技術:
```
可接受任意數量參數的函數
只接受關鍵字參數的函數
給函數參數增加元資訊
返回多個值的函數
定義有預設參數的函數
定義匿名或內聯函數
匿名函數捕獲變數值
減少可調用物件的參數個數
將單方法的類轉換為函數
帶額外狀態資訊的回呼函數
內聯回呼函數
訪問閉包中定義的變數
```
