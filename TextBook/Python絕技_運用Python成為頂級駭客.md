
```
>>> from sklearn.neighbors import KNeighborsClassifier
>>> knn = KNeighborsClassifier(n_neighbors=5, p=2,
... metric='minkowski')
>>> knn.fit(X_train_std, y_train)
>>> plot_decision_regions(X_combined_std, y_combined,
... classifier=knn, test_idx=range(105,150))
>>> plt.xlabel('petal length [standardized]')
>>> plt.ylabel('petal width [standardized]')
>>> plt.show()
```


- [NumPy](http://www.numpy.org) >= 1.12.1
- [SciPy](http://www.scipy.org) >= 0.19.0
- [scikit-learn](http://scikit-learn.org/stable/) >= 0.18.1
- [matplotlib](http://matplotlib.org) >= 2.0.2
- [pandas](http://pandas.pydata.org) >= 0.20.1

**Side Note:**  "IPython Notebook" recently became the "[Jupyter Notebook](<http://jupyter.org>)"; Jupyter is an umbrella project that aims to support other languages in addition to Python including Julia, R, and many more. Don't worry, though, for a Python user, there's only a difference in terminology (we say "Jupyter Notebook" now instead of "IPython Notebook").

The Jupyter notebook can be installed as usually via pip.

    $ pip install jupyter notebook

Alternatively, we can use the Conda installer if we have Anaconda or Miniconda installed:

    $ conda install jupyter notebook

To open a Jupyter notebook, we `cd` to the directory that contains your code examples, e.g,.


    $ cd ~/code/python-machine-learning-book

and launch `jupyter notebook` by executing

    $ jupyter notebook

```
第1章 入門 1
引言：使用Python進行滲透測試 1
準備開發環境 2
安裝協力廠商庫 2
Python解釋與Python交互 5
Python語言 6
變數 6
字串 7
List（列表） 7
詞典 8
網路 9
條件選擇語句 9
異常處理 10
函數 11
反覆運算 13
檔輸入/輸出 15
sys模組 16
OS模組 17
第一個Python程式 19
第一個程式的背景材料：布穀蛋 19
第一個程式：UNIX口令破解機 20
第二個程式的背景材料：度惡為善 22
第二個程式：一個Zip檔口令破解機 23
本章小結 27
參考文獻 28


第2章 用Python進行滲透測試 29
引言：Morris蠕蟲現在還有用嗎 29
編寫一個埠掃描器 30
TCP全連接掃描 30
抓取應用的Banner 32
執行緒掃描 34
使用NMAP埠掃描代碼 36
用Python構建一個SSH僵屍網路 38
用Pexpect與SSH交互 39
用Pxssh暴力破解SSH密碼 42
利用SSH中的弱私密金鑰 45
構建SSH僵屍網路 49
利用FTP與Web批量抓“肉機” 52
用Python構建匿名FTP掃描器 53
使用Ftplib暴力破解FTP用戶口令 54
在FTP伺服器上搜索網頁 55
在網頁中加入惡意注入代碼 56
整合全部的攻擊 58
Conficker，為什麼努力做就夠了 62
使用Metasploit攻擊Windows SMB服務 64
編寫Python腳本與Metasploit交互 65
暴力破解口令，遠端執行一個進程 67
把所有的代碼放在一起，構成我們自己的Conficker 67
編寫你自己的0day概念驗證代碼 70
基於棧的緩衝區溢位攻擊 70
添加攻擊的關鍵元素 71
發送漏洞利用代碼 72
匯總得到完整的漏洞利用腳本 73
本章小結 75
參考文獻 75


第3章 用Python進行取證調查 77
引言：如何通過電子取證解決BTK兇殺案 77
你曾經去過哪裡？——在註冊表中分析無線訪問熱點 78
使用WinReg讀取Windows註冊表中的內容 79
使用Mechanize把MAC位址傳給Wigle 81
用Python恢復被刪入回收站中的內容 85
使用OS模組尋找被刪除的檔/資料夾 85
用Python把SID和用戶名關聯起來 86
中繼資料 88
使用PyPDF解析PDF檔中的中繼資料 88
理解Exif中繼資料 90
用BeautifulSoup下載圖片 91
用Python的影像處理庫讀取圖片中的Exif中繼資料 92
用Python分析應用程式的使用記錄 95
理解Skype中的SQLite3資料庫 95
使用Python和SQLite3自動查詢Skype的資料庫 97
用Python解析火狐流覽器的SQLite3資料庫 103
用Python調查iTunes的手機備份 111
本章小結 116
參考文獻 116


第4章 用Python分析網路流量 119
引言：“極光”行動以及為什麼明顯的跡象會被忽視 119
IP流量將何去何從？——用Python回答 120
使用PyGeoIP關聯IP位址和物理位置 121
使用Dpkt解析包 121
使用Python畫穀歌地圖 125
“匿名者”真能匿名嗎？分析LOIC流量 128
使用Dpkt發現下載LOIC的行為 128
解析Hive伺服器上的IRC命令 130
即時檢測DDoS攻擊 131
H.D.Moore是如何解決五角大樓的麻煩的 136
理解TTL欄位 136
用Scapy解析TTL欄位的值 138
“風暴”（Storm）的fast-flux和Conficker的domain-flux 141
你的DNS知道一些不為你所知的嗎？ 142
使用Scapy解析DNS流量 143
用Scapy找出fast-flux流量 144
用Scapy找出Domain Flux流量 145
Kevin Mitnick和TCP序號預測 146
預測你自己的TCP序號 147
使用Scapy製造SYN泛洪攻擊 148
計算TCP序號 148
偽造TCP連接 150
使用Scapy愚弄入侵偵測系統 153
本章小結 159
參考文獻 159


第5章 用Python進行無線網路攻擊 161
引言：無線網路的（不）安全性和冰人 161
搭建無線網路攻擊環境 162
用Scapy測試無線網卡的嗅探功能 162
安裝Python藍牙包 163
綿羊牆——被動竊聽無線網路中傳輸的秘密 165
使用Python規則運算式嗅探信用卡資訊 165
嗅探賓館住客 168
編寫穀歌鍵盤記錄器 171
嗅探FTP登錄口令 174
你帶著筆記型電腦去過哪裡？Python告訴你 176
偵聽802.11 Probe請求 176
尋找隱藏網路的802.11信標 177
找出隱藏的802.11網路的網路名 178
用Python截取和監視無人機 179
截取資料包，解析協定 179
用Scapy製作802.11資料幀 181
完成攻擊，使無人機緊急迫降 184
探測火綿羊 186
理解Wordpress的會話cookies 187
牧羊人——找出Wordpress Cookie重放攻擊 188
用Python搜尋藍牙 190
截取無線流量，查找（隱藏的）藍牙設備位址 192
掃描藍牙RFCOMM通道 195
使用藍牙服務發現協定 196
用Python ObexFTP控制印表機 197
用Python利用手機中的BlueBug漏洞 197
本章小結 199
參考文獻 199

第6章 用Python刺探網路 201
引言：當今的社會工程 201
攻擊前的偵察行動 202
使用Mechanize庫上網 202
匿名性——使用代理伺服器、User-Agent及cookie 203
把代碼集成在Python類的AnonBrowser中 206
用anonBrowser抓取更多的Web頁面 208
用Beautiful Soup解析Href連結 209
用Beautiful Soup映射圖像 211
研究、調查、發現 213
用Python與穀歌API交互 213
用Python解析Tweets個人主頁 216
從推文中提取地理位置資訊 218
用規則運算式解析Twitter用戶的興趣愛好 220
匿名電子郵件 225
批量社工 226
使用Smtplib給目標物件發郵件 226
用smtplib進行網路釣魚 227
本章小結 230
參考文獻 231

第7章 用Python實現免殺 233
引言：火焰騰起！ 233
免殺的過程 234
免殺驗證 237
本章小結 243
參考文獻 243
編輯推薦
關於Python的書雖然已有不少，但從安全從業者角度全方位剖析Python的書籍幾乎沒有，《Python絕技：運用Python成為**駭客》填補了這個
```
