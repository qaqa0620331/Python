# Python standard library

>* Python有許多內建的函式庫(可以寫成幾本書)
>* 不需安裝即可使用(第三方套件需要安裝==>Tensorflow/PyTorch)
>* 使用時請import載入相關函式庫即可
>* 

# hashlib函式庫/套件/模組

>* hashlib是python專門提供hash演算法的函式庫，包括md5, sha1, sha224, sha256, sha384, sha512等演算法

```
>>> import hashlib
>>> m = hashlib.md5()
>>> m.update(b"Nobody inspects")
>>> m.update(b" the spammish repetition")
>>> m.digest()
    b'\\xbbd\\x9c\\x83\\xdd\\x1e\\xa5\\xc9\\xd9\\xde\\xc9\\xa1\\x8d\\xf0\\xff\\xe9'

More condensed:
>>> hashlib.sha224(b"Nobody inspects the spammish repetition").hexdigest()
```

##### Python2的範例==>作業:改成Python3
```
import hashlib

data = "I am your greatteacher"
print hashlib.md5(data).hexdigest()
print hashlib.sha1(data).hexdigest()
print hashlib.sha224(data).hexdigest()
print hashlib.sha256(data).hexdigest()
print hashlib.sha384(data).hexdigest()
print hashlib.sha512(data).hexdigest()
```

```
#!/usr/bin/python3
import hashlib
m = hashlib.md5()
data = "I am your greatteacher"

# 先將資料編碼，再更新 MD5 雜湊值
m.update(data.encode("utf-8"))

h = m.hexdigest()
print(h)
```

##### 只是小改變==>答案就會不一樣
```
import hashlib

a = "I am your greatteacher"
print hashlib.md5(a).hexdigest()

b = "I am your ggreatteacher"
print hashlib.md5(b).hexdigest()

c = "I am your Greatteacher"
print hashlib.md5(c).hexdigest()
```

##### 把你的機密檔案Hash
```
#!/usr/bin/env python

import hashlib
import sys

def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: %s file' % sys.argv[0])

    filename = sys.argv[1]
    m = hashlib.md5()
    with open(filename, 'rb') as fp:
        while True:
            blk = fp.read(4096) # 4KB per block
            if not blk: break
            m.update(blk)
    print(m.hexdigest(), filename)

if __name__ == '__main__':
    main()
```
執行看看:

python3 test4.py

python3 test4.py test1.py

# Pillow函式庫/套件/模組(Python3)

##### PIL(Python2) vs  Pillow(Python3)

>* PIL(Python Imaging Library)是Python平臺上的影像處理標準庫。PIL功能非常強大，但API卻非常簡單易用。PIL僅支持到Python 2.7
>* 一群志願者在PIL的基礎上創建了相容的版本，名字叫Pillow，支持最新Python 3.x，加入了許多新特性

安装Pillow==>sudo pip install pillow

##### 應用範例:圖片模糊特效
```
from PIL import Image, ImageFilter

# 打開一個jpg影像檔，注意是在當前路徑:
im = Image.open('cat.png')

# 應用模糊濾鏡filter():
im2 = im.filter(ImageFilter.BLUR)

# 將模糊濾鏡後的圖片存檔:
im2.save('cat_blur.jpg', 'jpeg')
```

##### 應用範例:圖片縮放特效
```
from PIL import Image

# 打開一個jpg影像檔，注意是當前路徑:
im = Image.open('cat.png')

# 獲得圖像尺寸:
w, h = im.size
print('Original image size: %sx%s' % (w, h))

# 縮放到50%:
im.thumbnail((w//2, h//2))
print('Resize image to: %sx%s' % (w//2, h//2))

# 把縮放後的圖像用jpeg格式保存:
im.save('cat_thumbnail.jpg', 'jpeg')
```

##### 應用範例:產生字母驗證碼圖片

先到底下下載自型 https://github.com/JotJunior/PHP-Boleto-ZF2/blob/master/public/assets/fonts/arial.ttf

存到目錄底下

```
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random

# 隨機字母:
def rndChar():
    return chr(random.randint(65, 90))

# 隨機顏色1:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 隨機顏色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

# 240 x 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))

# 創建Font對象:
font = ImageFont.truetype('rial.ttf', 36)

# 創建Draw對象:
draw = ImageDraw.Draw(image)

# 填充每個圖元:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())

# 輸出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())

# 模糊:
image = image.filter(ImageFilter.BLUR)

image.save('code.jpg', 'jpeg')

```


## Argparse函式庫/套件/模組

>* https://docs.python.org/3.6/howto/argparse.html
>* https://docs.python.org/2.7/howto/argparse.html

範例:
```
from argparse import ArgumentParser

parser1 = ArgumentParser()
parser2 = ArgumentParser(prog="my_example")
parser3 = ArgumentParser(usage="usage")
parser4 = ArgumentParser(description="a simple demo of argparse")
parser5 = ArgumentParser(epilog="see the doc: https://docs.python.org/3/library/argparse.html")

parser1.print_help()
parser2.print_help()
parser3.print_help()
parser4.print_help()
parser5.print_help()
```
-h 這個選擇性參數 (optional argument) 是自動產生的

##### argparse.ArgumentParser類別
```
class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], 
formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, 
conflict_handler='error', add_help=True, allow_abbrev=True)
```
>* Create a new ArgumentParser object. 
>* All parameters should be passed as keyword arguments. 

參數說明
>* prog - The name of the program (default: sys.argv[0])
>* usage - The string describing the program usage (default: generated from arguments added to parser)
>* description - Text to display before the argument help (default: none)
>* epilog - Text to display after the argument help (default: none)
>* parents - A list of ArgumentParser objects whose arguments should also be included
>* formatter_class - A class for customizing the help output
>* prefix_chars - The set of characters that prefix optional arguments (default: ‘-‘)
>* fromfile_prefix_chars - The set of characters that prefix files from which additional arguments should be read (default: None)
>* argument_default - The global default value for arguments (default: None)
>* conflict_handler - The strategy for resolving conflicting optionals (usually unnecessary)
>* add_help - Add a -h/--help option to the parser (default: True)
>* allow_abbrev - Allows long options to be abbreviated if the abbreviation is unambiguous. (default: True)

## 使用 ArgumentParser.add_argument() 加入想解析的參數

參數基本上分兩種，一種是位置參數 (positional argument)，另一種就是選擇性參數 (optional argument)，主要差別在於參數指定方式的不同
>* positional argument ==>根據它出現的位置來指定它的值是多少
>* optional argument 與位置無關，是根據前綴來指定 (底下範例 -o 或 --optional-arg 後面跟著是誰就指定是誰)。
```
from __future__ import print_function
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("pos1", help="positional argument 1")
parser.add_argument("-o", "--optional-arg", help="optional argument", dest="opt", default="default")

args = parser.parse_args()
print("positional arg:", args.pos1)
print("optional arg:", args.opt)
``` 

執行程式(不帶參數)==>python3 test2.py 

執行畫面
```
usage: test2.py [-h] [-o OPT] pos1
test2.py: error: the following arguments are required: pos1
```

執行程式==>python3 test2.py -h

執行畫面
```
usage: test2.py [-h] [-o OPT] pos1

positional arguments:
  pos1                  positional argument 1

optional arguments:
  -h, --help            show this help message and exit
  -o OPT, --optional-arg OPT
                        optional argument

```

執行程式(帶多個參數)==>python3 test2.py hello 

執行畫面
```
positional arg: hello
optional arg: default
```

執行程式(帶多個參數)==>python3 test2.py hello -o mydeargreatteacher

執行畫面
```
positional arg: hello
optional arg: mydeargreatteacher
```
##### add_argument()函式:
```
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, 
default][, type][, choices][, required][, help][, metavar][, dest])
```
>* Define how a single command-line argument should be parsed. 
>* Each parameter has its own more detailed description below, but in short they are:

參數說明
>* name or flags - Either a name or a list of option strings, e.g. foo or -f, --foo.
>* action - The basic type of action to be taken when this argument is encountered at the command line.
>* nargs - The number of command-line arguments that should be consumed.
>* const - A constant value required by some action and nargs selections.
>* default - The value produced if the argument is absent from the command line.
>* type - The type to which the command-line argument should be converted.
>* choices - A container of the allowable values for the argument.
>* required - Whether or not the command-line option may be omitted (optionals only).
>* help - A brief description of what the argument does.
>* metavar - A name for the argument in usage messages.
>* dest - The name of the attribute to be added to the object returned by parse_args().

##### parse_args()函式method的參數:只有兩個參數
```
ArgumentParser.parse_args(args=None, namespace=None)
```
Convert argument strings to objects and assign them as attributes of the namespace. Return the populated namespace.

Previous calls to add_argument() determine exactly what objects are created and how they are assigned. See the documentation for add_argument() for details.

參數說明
>* args - List of strings to parse. The default is taken from sys.argv.
>* namespace - An object to take the attributes. The default is a new empty Namespace object.

# 比較有趣的範例
```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", type=int, help="display a square of a given number")
parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity")

args = parser.parse_args()

answer = args.square**2

if args.verbosity == 2:
    print("the square of {} equals {}".format(args.square, answer))
elif args.verbosity == 1:
    print ("{}^2 == {}".format(args.square, answer))
else:
    print(answer)
```
```
python3 test3.py 2
4

python3 test3.py 2 -v 1
2^2 == 4

python3 test3.py 2 -v 2
the square of 2 equals 4

python3 test3.py 2 -v 33
答案會是甚麼??
```

#### 作業:將python 2.7的範例改成Python3的範例
https://docs.python.org/2.7/howto/argparse.html
