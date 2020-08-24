



class bird(object): # 定义鸟这个类

    def __init__(self, name): # 定义类时需要传入的参数，这里指创建实例时需要传入name参数
        self.name = name # 将参数赋值给self.name，成为属性，后面定义方法时会调用

    def move(self): # 每个类实现一次move方法
        print("The bird named" ,self.name ,"is flying")


class dog(object): # 定义狗这个类

    def __init__(self, name):
        self.name = name

    def move(self):
        print("The dog named" ,self.name ,"is running")

class fish(object): # 定义鱼这个类

    def __init__(self, name):
        self.name = name

    def move(self):
        print("The fish named" ,self.name ,"is swimming")



bob = bird("Bob") # 给bob这个变量（对象）传入“姓名”name参数
john = bird("John") # 产生两个bird的实例来对比
david = dog("David")
fabian = fish("Fabian")

bob.move() #  1 标号（方便文字说明）
john.move() # 2
david.move() # 3
fabian.move() # 4