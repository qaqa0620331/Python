# 教科書

```
Foundations of Python Network Programming
https://www.apress.com/br/book/9781430230038
https://github.com/Apress/foundations-of-python-network-programming
```

```
Python Network Programming Cookbook - Second Edition
Pradeeban Kathiravelu, Dr. M. O. Faruque Sarker
August 2017
https://www.packtpub.com/networking-and-servers/python-network-programming-cookbook-second-edition
https://github.com/lovexiaov/learn-python/tree/master/python-network-programming-cookbook
```

# 使用 socket 模組

>* 使用python2內建socket模組
>* https://docs.python.org/2/library/socket.html


```
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import socket #載入python2內建socket模組

hostname = 'maps.google.com'
addr = socket.gethostbyname(hostname)  #使用python2內建socket模組的gethostbyname()方法

print 'The address of', hostname, 'is', addr
```

### 使用getaddrinfo函式

**語法**:socket.getaddrinfo(host, port[, family[, socktype[, proto[, flags]]]])
