
# Object Oriented Programming 物件導向程式設計(OOP)

>* 針對大型軟體設計提出，目的是使軟體設計更加靈活、改良與支援程式碼和設計的重用性，程式碼具有更好的可讀性和可擴展性，以及大幅降低軟體發展的難度等。 

>* 物件導向程式設計特色:封裝(Encapsulation)+繼承(Inheritance)+多型(Polymorphism)
>* 物件導向程式設計特色:封裝
>* 將資料以及相關的操作封裝在一起， 以便組成一個相互依存、不可分割的整體(即類別class與物件object)， 不同物件之間透過訊息機制來通訊或者同步。 
>* 針對相同類型的物件(instance)進行分頭、抽象，得出共同的特體而形成類別(class) 

物件導向程式設計的關鍵， 就是如何合理地定義類別(class) ，並且祖捕多個類別(class) 之間的關 系。

### Python與物件導向

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
### object 類別

>* Python中的類別都是來自object類別 。
>* 當定義類別沒有指定繼承時，則此類別預設的父類別就是object類別

>* 所有定義於object的方法都是特別的方法，它們都有兩個前導底線和兩個後繼的底線。

### 類別方法
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

### 多重繼承(multiple inheritance)




繼承與多型
