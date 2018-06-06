# 教科書

```
Foundations of Python Network Programming
https://www.apress.com/br/book/9781430230038
https://github.com/Apress/foundations-of-python-network-programming

Python网络编程 第3版

```


```
Python Network Programming Cookbook - Second Edition
Pradeeban Kathiravelu, Dr. M. O. Faruque Sarker
August 2017
https://www.packtpub.com/networking-and-servers/python-network-programming-cookbook-second-edition
https://github.com/lovexiaov/learn-python/tree/master/python-network-programming-cookbook
```

```
Python Penetration Testing Cookbook
Rejah Rehim
November 2017
https://www.packtpub.com/networking-and-servers/python-penetration-testing-cookbook
https://github.com/PacktPublishing/Python-Penetration-Testing-Cookbook
```

# 使用 socket 模組

>* 使用python2內建socket模組
>* https://docs.python.org/2/library/socket.html

### 使用python2內建socket模組的gethostbyname()方法
```
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import socket #載入python2內建socket模組

hostname = 'maps.google.com'
addr = socket.gethostbyname(hostname)  

print 'The address of', hostname, 'is', addr
```

### 使用getaddrinfo函式

**語法**:socket.getaddrinfo(host, port[, family[, socktype[, proto[, flags]]]])


Python Network Programming Cookbook - Second Edition,Pradeeban Kathiravelu, Dr. M. O. Faruque Sarker

# 使用urllib2 

>* Python Penetration Testing Cookbook,Rejah Rehim,November 2017

### download_image
```
import urllib2
import re
from os.path import basename
from urlparse import urlsplit

url = 'https://www.packtpub.com/'

response = urllib2.urlopen(url)
source = response.read()
file = open("packtpub.txt", "w")
file.write(source)
file.close()

patten = '(http)?s?:?(\/\/[^"]*\.(?:png|jpg|jpeg|gif|png|svg))'
for line in open('packtpub.txt'):
    for m in re.findall(patten, line):
        print('https:' + m[1])
        fileName = basename(urlsplit(m[1])[2])
        print(fileName)
        try:
            img = urllib2.urlopen('https:' + m[1]).read()
            file = open(fileName, "w")
            file.write(img)
            file.close()
        except:
            pass
        break
 ```      
 
 ``` 
import urllib.request
import urllib.parse
import re
from os.path import basename

url = 'https://www.packtpub.com/'

response = urllib.request.urlopen(url)
source = response.read()
file = open("packtpub.txt", "wb")
file.write(source)
file.close()

patten = '(http)?s?:?(\/\/[^"]*\.(?:png|jpg|jpeg|gif|png|svg))'

for line in open('packtpub.txt'):
    for m in re.findall(patten, line):
        print('https:' + m[1])
        fileName = basename(urllib.parse.urlsplit(m[1])[2])
        print(fileName)
        try:
            img = urllib.request.urlopen('https:' + m[1]).read()
            file = open(fileName, "wb")
            file.write(img)
            file.close()
        except:
            pass
        break 
 ``` 
 
 ``` 
import urllib.request
import urllib.parse
import re
from os.path import basename

url = 'https://www.packtpub.com/'
queryString = 'all?search=&offset='

for i in range(0, 200, 12):
    query = queryString + str(i)
    url += query
    print(url)
    response = urllib.request.urlopen(url)
    source = response.read()
    file = open("packtpub.txt", "wb")
    file.write(source)
    file.close()

    patten = '(http)?s?:?(\/\/[^"]*\.(?:png|jpg|jpeg|gif|png|svg))'
    for line in open('packtpub.txt'):
        for m in re.findall(patten, line):
            print('https:' + m[1])
            fileName = basename(urllib.parse.urlsplit(m[1])[2])
            print(fileName)
            request = 'https:' + urllib.parse.quote(m[1])
            img = urllib.request.urlopen(request).read()
            file = open(fileName, "wb")
            file.write(img)
            file.close()

            break
```

```
import urllib.request
import urllib.parse
import re
from os.path import basename

url = 'https://www.packtpub.com/'

response = urllib.request.urlopen(url)
source = response.read()
file = open("packtpub.txt", "wb")
file.write(source)
file.close()

patten = '(http)?s?:?(\/\/[^"]*\.(?:png|jpg|jpeg|gif|png|svg))'
for line in open('packtpub.txt'):
    for m in re.findall(patten, line):
        print('https:' + m[1])
        fileName = basename(urllib.parse.urlsplit(m[1])[2])
        print(fileName)
        request = 'https:' + urllib.parse.quote(m[1])
        img = urllib.request.urlopen(request).read()
        file = open(fileName, "wb")
        file.write(img)
        file.close()

        break 
```
