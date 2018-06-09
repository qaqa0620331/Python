
Python網絡爬蟲從入門到實踐
簡體中文/唐松, 陳智銓/機械工業出版社出版日期：2017-09-05

https://github.com/Santostang/PythonScraping

```

第1章網絡爬蟲入門
1.1為什麼要學網絡爬蟲
1.1.1網絡爬蟲能帶來什麼好處
1.1.2能從網絡上爬取什麼數據
1.1.3應不應該學爬蟲
1.2網絡爬蟲是否合法
1.2.1 Robots協議
1.2.2網絡爬蟲的約束
1.3網絡爬蟲的基本議題
1.3.1 Python爬蟲的流程
1.3.2三個流程的技術實現
```
```
2章編寫第1個網絡爬蟲
2.1搭建Python平台
2.1.1 Python的安裝
2.1.2使用pip安裝第三方庫
2.1.3使用編譯器Jupyter編程
2.2 Python使用入門
2.2.1基本命令
2.2.2數據類型
2.2.3條件語句和循環語句
2.2.4函數
2.2 .5面向對象編程
```
### 2.3編寫第1個簡單的爬蟲

2.3.1 第一步：獲取頁面
```
#!/usr/bin/python
# coding: utf-8

import requests
link = "http://www.santostang.com/"
headers = {'User-Agent' : 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'} 

r = requests.get(link, headers= headers)
print (r.text)
```
2.3.2 第二步：提取需要的資料
```

#!/usr/bin/python#!/usr/ 
# coding: utf-8

import requests
from bs4 import BeautifulSoup    

link = "http://www.santostang.com/"
headers = {'User-Agent' : 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'} 
r = requests.get(link, headers= headers)

soup = BeautifulSoup(r.text, "html.parser")     
title = soup.find("h1", class_="post-title").a.text.strip()
print (title)

```

```
import requests
from bs4 import BeautifulSoup   #從bs4這個第三方函士庫中導入BeautifulSoup

link = "http://www.santostang.com/"
headers = {'User-Agent' : 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'} 
r = requests.get(link, headers= headers)

soup = BeautifulSoup(r.text, "html.parser")   #使用BeautifulSoup解析這段代碼
title = soup.find("h1", class_="post-title").a.text.strip()
print (title)

with open('title_test.txt', "a+") as f:
    f.write(title)
    f.close()
```

第3章靜態網頁抓取

第4章動態網頁抓取
第5章解析網頁
第6章數據存儲
第7章提升爬蟲的速度
第8章反爬蟲問題
第9章解決中文亂碼
第10章登錄與驗證碼處理
第11章服務器採集
第12章分佈式爬蟲
第13章爬蟲實踐一：維基百科
第14章爬蟲實踐二：知乎Live 
第15章爬蟲實踐三：百度地圖API 
第16章爬蟲實踐四：餐廳點評
```
