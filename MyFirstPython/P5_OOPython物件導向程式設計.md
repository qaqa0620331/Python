
# Object Oriented Programming 物件導向程式設計(OOP)

>* 針對大型軟體設計提出，目的是使軟體設計更加靈活、改良與支援程式碼和設計的重用性，程式碼具有更好的可讀性和可擴展性，以及大幅降低軟體發展的難度等。 

>* 物件導向程式設計特色:封裝(Encapsulation)+繼承(Inheritance)+多型(Polymorphism)
>* 物件導向程式設計特色:封裝
>* 將資料以及相關的操作封裝在一起， 以便組成一個相互依存、不可分割的整體(即類別class與物件object)， 不同物件之間透過訊息機制來通訊或者同步。 
>* 針對相同類型的物件(instance)進行分頭、抽象，得出共同的特體而形成類別(class) 

物件導向程式設計的關鍵， 就是如何合理地定義類別(class) ，並且祖捕多個類別(class) 之間的關 系。

### Python與物件導向

The Python Language Reference
>* https://docs.python.org/3.6/reference/index.html
>* https://docs.python.org/2.7/reference/index.html

>* Python 是真正物件導向式的高階動態程式語言， 完全支援物件導向的基本功能， 倒如封裝、繼承、多型，以及對基礎類別(class) 方法的覆蓋或重寫。 
>* Python 中的一切內容都可稱為物件，包括函數等。 
>* 建立類別(class)時，以變數形式表示物件特徵的成員稱為資料成員(attribute)，以函數形式表示物件行為的成員稱為成員方法(method)， 資料成員和成員方法前稱為錯別的成員。


# 類別class與物件object

定義class類別:Circle
>* 成員方法(method)[又稱動作(actions)]:定義物件的行為(behavior)。可以在物件上呼叫方法，就等同在物件執行某個動作
>* 一個圓形物件可呼叫getArea() 來取得面積，呼叫getPerimeter() 來取得周長。
>* 所有的方法(包括建構子)都有定義的第一個參數是self ，self參數是自動設定參考到剛被建立的物件來呼叫實例方法(instance methods) 。

```
import math 

class Circle:
    # Construct a circle object 
    def __init__(self, radius = 11):#建構子(constrictor)
        self.radius = radius

    def getPerimeter(self):
        return 2 * self.radius * math.pi

    def getArea(self):
        return self.radius * self.radius * math.pi
          
    def setRadius(self, radius):
        self.radius = radius
```

>* class類別:Circle
>* object物件:利用Circle類別建立三個物件circle1,circle2,circle3
>* 物件成員存取運算子(object member access operator)===>點運算子( . ):存取物件的資料項目與呼叫物件的方法
```
from Circle import Circle

def main():
    # Create a circle with radius 11
    circle1 = Circle()
    print("The area of the circle of radius",circle1.radius, "is", circle1.getArea())

    # Create a circle with radius 215
    circle2 = Circle(215)
    print("The area of the circle of radius", circle2.radius, "is", circle2.getArea())

    # Create a circle with radius 325
    circle3 = Circle(325)
    print("The area of the circle of radius",circle3.radius, "is", circle3.getArea())

    # Modify circle radius
    circle2.radius = 400
    print("The area of the circle of radius",circle2.radius, "is", circle2.getArea())

main() # Call the main function
```

```
class Shark:
    def __init__(self, name):
        self.name = name

    def swim(self):
        print(self.name + " is swimming.")

    def be_awesome(self):
        print(self.name + " is being awesome.")

def main():
    sammy = Shark("Sammy") #利用類別的建構子產生物件
    sammy.be_awesome()
    stevie = Shark("Stevie")#利用類別的建構子產生物件
    stevie.swim()

if __name__ == "__main__":
  main()
```

python shark.py

### UML類別圖

### 作業:完成教科書的練習

### datetime 類別的練習

>* https://docs.python.org/2/library/datetime.html
>* https://docs.python.org/3/library/datetime.html
>* 有空看看https://github.com/python/cpython/blob/3.6/Lib/datetime.py
```
from datetime import datetime
d = datetime.now()
print("Current year is " + str(d.year))
print("Current month is " + str(d.month))
print("Current day of month is " + str(d.day))
print("Current hour is " + str(d.hour))
print("Current minute is " + str(d.minute))
print("Current second is " + str(d.second))
```

### 資料隱藏(data hiding)

>* 為了防止直接修改資料項目，有一方法就是不要讓客戶端直接存取資料項目。
>* 可經由定義私有資料項目(private data fields)來完成。
>* 在Python定義私有的資料項目是在其前面加兩個底線，同理也可以此方式定義私有的方法。

```
import math 

class Circle:
    # Construct a circle object 
    def __init__(self, radius = 1):
        self.__radius = radius

    def getRadius(self):
        return self.__radius

    def getPerimeter(self):
        return 2 * self.__radius * math.pi

    def getArea(self):
        return self.__radius * self.__radius * math.pi
```



# 繼承(inheritance):父類別與子類別   

類別與類別之間的關係 
>* 關連 (association):是一般的二元關係，用來描述兩類別之間的活動。 學生----課程------老師
>* 聚合(aggregation):關連的特殊格式，用以表示兩物件之間所有權的關係。聚合的關係通常以聚合類別的資料成員來表示    Name---student ---Address
>* 合成(composition):一個物件可能被許多聚合物件所擁有，如果一個物件只被一個聚合物件獨家擁有，則它們的關係就是合成(composition)
>* 繼承(inheritance)


子類別繼承父類別:
>* 子類別就自動擁有父類別的變數與函式。
>* 子類別==衍生類別(derived class)或子類別(child class)
>* 父類別==基礎類別(base class)或雙親類別(parent class)

父類別:GeometricObject.py
```
class GeometricObject:
    def __init__(self, color = "green", filled = True):
        self.__color = color
        self.__filled = filled

    def getColor(self):
        return self.__color

    def setColor(self, color):
        self.__color = color

    def isFilled(self):
        return self.__filled

    def setFilled(self, filled):
        self.__filled = filled
  
    def __str__(self):
        return "color: " + self.__color + \
            " and filled: " + str(self.__filled)
```
 
CircleFromGeometricObject.py ==>子類別:Circle  繼承 父類別:GeometricObject
```
from GeometricObject import GeometricObject
import math

class Circle(GeometricObject):# 子類別:Circle  繼承 父類別:GeometricObject
    def __init__(self, radius):
        super().__init__() # 呼叫父類別的建構子
        self.__radius = radius

    def getRadius(self):
        return self.__radius

    def setRadius(self, radius):
        self.__radius = radius

    def getArea(self):
        return self.__radius * self.__radius * math.pi
  
    def getDiameter(self):
        return 2 * self.__radius
  
    def getPerimeter(self):
        return 2 * self.__radius * math.pi

    def printCircle(self):
        print(self.__str__() + " radius: " + str(self.__radius))
```

RectangleFromGeometricObject.py ==>子類別:Rectangle  繼承 父類別:GeometricObject
```
from GeometricObject import GeometricObject

class Rectangle(GeometricObject): #子類別:Rectangle  繼承 父類別:GeometricObject
    def __init__(self, width = 1, height = 1): 
        super().__init__()  # 呼叫父類別的建構子
        self.__width = width
        self.__height = height

    def getWidth(self):
        return self.__width

    def setWidth(self, width):
        self.__width = width

    def getHeight(self):
        return self.__height

    def setHeight(self, height):
        self.__height = self.__height

    def getArea(self):
        return self.__width * self.__height

    def getPerimeter(self):
        return 2 * (self.__width + self.__height)
```

主測試程式:
```
from CircleFromGeometricObject import Circle
from RectangleFromGeometricObject import Rectangle

def main():
    circle = Circle(1.5)
    print("A circle", circle)
    print("The radius is", circle.getRadius())
    print("The area is", circle.getArea())
    print("The diameter is", circle.getDiameter())
    
    rectangle = Rectangle(2, 4)
    print("\nA rectangle", rectangle)
    print("The area is", rectangle.getArea())
    print("The perimeter is", rectangle.getPerimeter())

main() # Call the main function

```
### object 類別(老祖宗)與magic method(魔術方法)/special method(特別方法)

>* Python中的類別都是來自object類別 。
>* 當定義類別沒有指定繼承時，則此類別預設的父類別就是object類別
```
Class GG:
   pass
   
>>> GG.__class__
>>> GG.__class__.__base__
```

>* 所有定義於object的方法都是magic method(魔術方法)/special method(特別方法)，它們都有兩個前導底線和兩個後繼的底線。

常用的special method 
>* https://docs.python.org/3.6/reference/datamodel.html#special-method-names
>* object.__new__(cls[, ...])  ==> __new__(cls,*args,**kwd) 
>* object.__init__(self[, ...]) ==> __init__(self,...)

```
 __init__(self,...)    建構子(constructor)::初始化物件，在創建新物件時調用
 __del__(self)     解構子(destrcutor)::釋放物件，在物件被刪除之前調用
 __new__(cls,*args,**kwd)     實例的生成操作
 __str__(self)     在使用print語句時被調用
 __getitem__(self,key)     獲取序列的索引key對應的值，等價於seq[key]
 __len__(self)     在調用內聯函數len()時被調用
 __cmp__(stc,dst)     比較兩個物件src和dst
 __getattr__(s,name)     獲取屬性的值
 __setattr__(s,name,value)     設置屬性的值
 __delattr__(s,name)     刪除name屬性
 __getattribute__()     __getattribute__()功能與__getattr__()類似
 __gt__(self,other)     判斷self物件是否大於other物件
 __lt__(slef,other)     判斷self物件是否小於other物件
 __ge__(slef,other)     判斷self物件是否大於或者等於other物件
 __le__(slef,other)     判斷self物件是否小於或者等於other物件
 __eq__(slef,other)     判斷self物件是否等於other物件
 __call__(self,*args)     把實例物件作為函式呼叫
```

>* 當一物件被建立，將自動呼叫 __new__() 方法。
>* 此方法接著呼叫 __init__() 方法，用以初始物件。
>* 正常情況下，應該只有覆寫 __init__() 方法來初始新類別的資料項目。 

>* __str__() 方法回傳描述此物件的字串。預設回傳的字串包括了物件名稱與物件在記憶體的位址(以16進位表示) 

### 類別方法(class method)
>* https://sites.google.com/site/zsgititit/home/python-cheng-shi-she-ji/python-lei-bie
>* 類別方法(class method)作用對象為類別，會影響整個類別，也會影響類別所產生的物件
>* 類別方法的第一個參數通常取名為cls，需在類別中的函式前一行使用裝飾器「@classmethod」

```
class Animal():
        count = 0  # 初始化類別變數count為0，
        
        """變數count是類別變數==>變數count在函式外部且前方沒有加上self，以此類別宣告的所有物件共用一個類別變數
        類別變數count使用「Animal.count」或「cls.count」進行存取。"""
        
        def __init__(self):
                 Animal.count += 1
        def kill(self):
                 Animal.count -= 1
        
        @classmethod  #定義類別方法
        def show_count(cls):
                 print('現在有',cls.count,'隻動物')
a = Animal()
Animal.show_count()
b = Animal()
Animal.show_count()
c = Animal()
Animal.show_count()
a.kill()
Animal.show_count()
```

### 靜態方法(static method)

>* 讓類別不需要建立物件，就可直接使用該類別的靜態方法
>* 需在類別中的函式前一行使用裝飾器「@staticmethod」。

```
class Welcome():
        @staticmethod  # 靜態方法(static method)
        def sayhello():
                 print('Hello Mydeargreatstudent')
                 
Welcome.sayhello()
```

### 方法覆寫(method overriding)

>* 子類別繼承從父類別的方法，有時可能會在子類別修改其繼承的父類別方法中的內容

範例:學生,修課課程與開課老師

建立Person類別[繼承object類別]
```
import datetime

class Person(object):
   def __init__(self, name):
     """Create a person"""
     self.name = name
     try:
       lastBlank = name.rindex(' ')
       self.lastName = name[lastBlank+1:]
     except:
       self.lastName = name
       self.birthday = None

   def getName(self):
     """Returns self's full name"""
     return self.name

   def getLastName(self):
     """Returns self's last name"""
     return self.lastName

   def setBirthday(self, birthdate):
     """Assumes birthdate is of type datetime.date
        Sets self's birthday to birthdate"""
     self.birthday = birthdate

   def getAge(self):
     """Returns self's current age in days"""
     if self.birthday == None:
        raise ValueError
        return (datetime.date.today() - self.birthday).days

   def __lt__(self, other):
     """Returns True if self's name is lexicographically
        less than other's name, and False otherwise"""
      if self.lastName == other.lastName:
         return self.name < other.name
      return self.lastName < other.lastName

   def __str__(self):
      """Returns self's name"""
      return self.name
```
>* Person類別複寫object類別定義的__init__,__str__與__lt__

```
me = Person('Mydeargreatteacher')
him = Person('Barack Hussein Obama')
her = Person('Madonna')

print him.getLastName()
him.setBirthday(datetime.date(1961, 8, 4))
her.setBirthday(datetime.date(1958, 8, 16))
print him.getName(), 'is', him.getAge(), 'days old'
```
```
pList = [me, him, her]
for p in pList:
print p

pList.sort()
for p in pList:
print p
```
現在再建立一個子類別繼承Person類別
```
class MITPerson(Person):
    nextIdNum = 0 #identification number

    def __init__(self, name):
       Person.__init__(self, name)
       self.idNum = MITPerson.nextIdNum #詳見底下說明
       MITPerson.nextIdNum += 1

    def getIdNum(self):
       return self.idNum

    def __lt__(self, other):
       return self.idNum < other.idNum
```

The method MITPerson.__init__ first invokes Person.__init__ to initialize the
inherited instance variable self.name. It then initializes self.idNum, an instance
variable that instances of MITPerson have but instances of Person do not.

The instance variable self.idNum is initialized using a class variable, nextIdNum,
that belongs to the class MITPerson, rather than to instances of the class. When
an instance of MITPerson is created, a new instance of nextIdNum is not created.
This allows __init__ to ensure that each instance of MITPerson has a unique
idNum.

### 多重繼承(multiple inheritance)

```
class Calculator:
  def calculate(self, expression):
      self.value = eval(expression)

class Talker:
    def talk(self):
        print('Hi, my value is', self.value)

class TalkingCalculator(Calculator, Talker):
    pass


tc = TalkingCalculator()
tc.calculate('1 + 2 * 3')
tc.talk()
```

# 多型

繼承的關係讓子類別繼承來自父類別的特性，再加上新的特性，因此子類別可以說是父類別的特例，所以每個子類別所產生的實體物件，也是它父類別的實體物件，但是相反就不是。例如每個圓形物件都是幾何物件，反過來說，不是每個幾何物件都是圓形物件，因此您可以將子類別的物件傳送給父類別的參數。 

```
from CircleFromGeometricObject import Circle
from RectangleFromGeometricObject import Rectangle

def main():
    # Display circle and rectangle properties
    c = Circle(4)
    r = Rectangle(1, 3)
    displayObject(c)
    displayObject(r)
    print("Are the circle and rectangle the same size?", isSameArea(c, r))

# Display geometric object properties 
def displayObject(g):
    print(g.__str__())

# Compare the areas of two geometric objects 
def isSameArea(g1, g2):
    return g1.getArea() == g2.getArea()

main() # Call the main function
```

# 類別與類別之間的關係 
>* 關連 (association):是一般的二元關係，用來描述兩類別之間的活動。 學生----課程------老師
>* 聚合(aggregation):關連的特殊格式，用以表示兩物件之間所有權的關係。聚合的關係通常以聚合類別的資料成員來表示    Name---student ---Address
>* 合成(composition):一個物件可能被許多聚合物件所擁有，如果一個物件只被一個聚合物件獨家擁有，則它們的關係就是合成(composition)
>* 繼承(inheritance)

```
class Course: 
    def __init__(self, courseName):
        self.__courseName = courseName
        self.__students = []
  
    def addStudent(self, student):
        self.__students.append(student)
  
    def getStudents(self):
        return self.__students

    def getNumberOfStudents(self):
        return len(self.__students)

    def getCourseName(self):
        return self.__courseName

    def dropStudent(student): 
        print("Left as an exercise")
```

```
from Course import Course

def main():
    course1 = Course("Data Structures")
    course2 = Course("Database Systems")

    course1.addStudent("Peter Jones")
    course1.addStudent("Brian Smith")
    course1.addStudent("Anne Kennedy")

    course2.addStudent("Peter Jones")
    course2.addStudent("Steve Smith")

    print("Number of students in course1:",
        course1.getNumberOfStudents())
    students = course1.getStudents()
    for student in students:
        print(student, end = ", ")
    
    print("\nNumber of students in course2:",
        course2.getNumberOfStudents())

main() # Call the main function

```


# 進階主題:decorator, iterator,

### decorator裝飾器

>* 一個裝飾器就是一個函數，它接受一個函數作為參數並返回一個新的函數
>* 內建的裝飾器如@staticmethod, @classmethod,@property 其運作原理都相同

底下兩段程式效果相同
```
class A:
   @classmethod
   def method(cls):
      pass
```
```
class B:
   def method(cls):
       pass
    method = classmethod(method)
```

#### 應用範例:自訂一個decorator為函式執行提供計時的額外功能

自訂一個decorator裝飾器
```
import time
from functools import wraps

def timethis(func):
'''
Decorator that reports the execution time.
'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs) #執行函式的運算
        end = time.time()
        print(func.__name__, end-start)
        return result
        return wrapper
```

使用裝飾器
```
>>> @timethis
... def countdown(n):
... '''
... Counts down
... '''
...    while n > 0:
...         n -= 1
...

>>> countdown(100000)
```

##### 一個裝飾器已經作用在一個函數上，你想撤銷這個裝飾功能，直接去訪問原始的未裝飾的那個函數。

see Python cookbook 11.3.1
>* http://python3-cookbook.readthedocs.io/zh_CN/latest/
>* https://github.com/dabeaz/python-cookbook

```
Python 錦囊妙計, 3/e (Python Cookbook, 3/e) 
David Beazley, Brian K. Jones 著、黃銘偉 譯
```

#### 如何定義一個帶參數的裝飾器

### Iterators 迭代器

使用for loop進行迭代運算(iterative)
```
>>> for i in [1, 2, 3, 4]:
...     print i,
...
```
```
>>> for c in "python":
...     print c
...
```

```
>>> for k in {"x": 1, "y": 2}:
...     print k
...
```

https://anandology.com/python-practice-book/iterators.html

```
>>> for line in open("a.txt"):
...     print line,
...

```

 上述都是iterable objects
 
 ```
 >>> ",".join(["a", "b", "c"])
'a,b,c'
>>> ",".join({"x": 1, "y": 2})
'y,x'
>>> list("python")
['p', 'y', 't', 'h', 'o', 'n']
>>> list({"x": 1, "y": 2})
['y', 'x']
```

#### The Iteration Protocol

>* 使用內建函式 built-in function iter 
>* takes an iterable object and returns an iterator.
>* Each time we call the next method on the iterator gives us the next element. 
>* If there are no more elements, it raises a StopIteration.

```
>>> x = iter([1, 2, 3])
>>> x
<listiterator object at 0x1004ca850>
>>> x.next()
1
>>> x.next()
2
>>> x.next()
3
>>> x.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
 ```
#### 實作一個Iterators類別功能和xrange一樣

Iterators are implemented as classes

```
 class yrange:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration()
```

>* The return value of __iter__ is an iterator. 
>* It should have a next method and raise StopIteration when there are no more elements.

執行結果==>
```
>>> y = yrange(3)
>>> y.next()
0
>>> y.next()
1
>>> y.next()
2
>>> y.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 14, in next
StopIteration
```

#### Many built-in functions accept iterators as arguments.

```
>>> list(yrange(5))
[0, 1, 2, 3, 4]
>>> sum(yrange(5))
10
```

### Generator產生器

>* A generator is a function that produces a sequence of results instead of a single value.
>* Generators simplifies creation of iterators.
>* a generator is also an iterator. You don’t have to worry about the iterator protocol.
```
def yrange(n):
    i = 0
    while i < n:
        yield i
        i += 1
```
```
>>> y = yrange(3)
>>> y
<generator object yrange at 0x401f30>
>>> y.next()
0
>>> y.next()
1
>>> y.next()
2
>>> y.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

仔細分析底下的程式執行
```
>>> def foo():
...     print "begin"
...     for i in range(3):
...         print "before yield", i
...         yield i
...         print "after yield", i
...     print "end"
...
>>> f = foo()
>>> f.next()
begin
before yield 0
0
>>> f.next()
after yield 0
before yield 1
1
>>> f.next()
after yield 1
before yield 2
2
>>> f.next()
after yield 2
end
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
>>>
```

### generator expression<<產生器>>

>* Generator Expressions are generator version of list comprehensions. 
>* They look like list comprehensions, but returns a generator back instead of a list

```
>>> a = (x*x for x in range(10))
>>> a
<generator object <genexpr> at 0x401f08>
>>> sum(a)
285
```

#### 可以使用 generator expressions 當作arguments to various functions that consume iterators.
```
>>> sum((x*x for x in range(10)))
285
```
#### 應用範例:找出(由小到大的)前十個直角三角形 pythogorian triplet

A triplet (x, y, z) is called pythogorian triplet if x*x + y*y == z*z.
```
>>> pyt = ((x, y, z) for z in integers() for y in xrange(1, z) for x in range(1, y) if x*x + y*y == z*z)
>>> take(10, pyt)
```
[(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17), (12, 16, 20), (15, 20, 25), (7, 24, 25), (10, 24, 26), (20, 21, 29)]


### 善用itertools(內建模組)

chain – chains multiple iterators together.
```
>>> it1 = iter([1, 2, 3])
>>> it2 = iter([4, 5, 6])
>>> itertools.chain(it1, it2)
[1, 2, 3, 4, 5, 6]
```

izip – iterable version of zip
```
>>> for x, y in itertools.izip(["a", "b", "c"], [1, 2, 3]):
...     print x, y
...
a 1
b 2
c 3
```

#### 如何反覆運算一個序列(list)的同時跟蹤正在被處理的元素索引(index)

see Python cookbook 11.3.1
```
>>> my_list = ['a', 'b', 'c']
>>> for idx, val in enumerate(my_list):
...     print(idx, val)

```

