'''
二十三种设计模式及python 实现
https://blog.csdn.net/weicao1990/article/details/79108193
'''


'''
Method(工厂方法)：
意图：
定义一个用于创建对象的接口，让子类决定实例化哪一个类。Factory Method 使一个类的实例化延迟到其子类。

适用性：
当一个类不知道它所必须创建的对象的类的时候。
当一个类希望由它的子类来指定它所创建的对象的时候。
当类将创建对象的职责委托给多个帮助子类中的某一个，并且你希望将哪一个帮助子类是代理者这一信息局部化的时候。

工厂方法介绍文章
https://segmentfault.com/a/1190000013053013
'''

class ChinaGetter:
    """A simple localizer a la gettext"""
    def __init__(self):
        self.trans = dict(dog=u"小狗", cat=u"小猫")

    def get(self, msgid):
        """We'll punt if we don't have a translation"""
        try:
            return self.trans[msgid]
        except KeyError:
            return str(msgid)

class EnglishGetter:
    """Simply echoes the msg ids"""
    def get(self, msgid):
        return str(msgid)

# 工厂方法
def get_localizer(language="English"):
    """The factory method"""
    languages = dict(English=EnglishGetter, China=ChinaGetter)
    return languages[language]()

# 示例运行
# Create our localizers
# e, g = get_localizer("English"), get_localizer("China")
# Localize some text
# for msgid in "dog parrot cat bear".split():
#     print(e.get(msgid), g.get(msgid))


'''
Abstract Factory(抽象工厂: 解决复杂对象创建问题)
意图：
提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。 

适用性：
一个系统要独立于它的产品的创建、组合和表示时。
一个系统要由多个产品系列中的一个来配置时。
当你要强调一系列相关的产品对象的设计以便进行联合使用时。
当你提供一个产品类库，而只想显示它们的接口而不是实现时。
'''

import random

class PetShop:
    """A pet shop"""
    def __init__(self, animal_factory=None):
        """pet_factory is our abstract factory.
        We can set it at will."""
        self.pet_factory = animal_factory

    def show_pet(self):
        """Creates and shows a pet using the
        abstract factory"""

        pet = self.pet_factory.get_pet()  #继承工厂类方法
        print("This is a lovely", str(pet))
        print("It says", pet.speak())
        print("It eats", self.pet_factory.get_food())

# 创建类对象
# Stuff that our factory makes
class Dog:
    def speak(self):
        return "woof"

    def __str__(self):
        return "Dog"

class Cat:
    def speak(self):
        return "meow"

    def __str__(self):
        return "Cat"

# 工厂类
# Factory classes
class DogFactory:
    def get_pet(self):
        return Dog()

    def get_food(self):
        return "dog food"

class CatFactory:
    def get_pet(self):
        return Cat()

    def get_food(self):
        return "cat food"

# Create the proper family
def get_factory():
    """Let's be dynamic!"""
    return random.choice([DogFactory, CatFactory])()

# 代码运行实例
# shop = PetShop()
# for i in range(3):
#     shop.pet_factory = get_factory()
#     shop.show_pet()
#     print("=" * 20)


'''
3. Builder（建造者）
意图：
将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。

适用性：
当创建复杂对象的算法应该独立于该对象的组成部分以及它们的装配方式时。
当构造过程必须允许被构造的对象有不同的表示时。
'''

# Director
class Director(object):
    def __init__(self):
        self.builder = None

    def construct_building(self):
        self.builder.new_building()
        self.builder.build_floor()
        self.builder.build_size()

    def get_building(self):
        return self.builder.building

# 抽象类用于被子类继承
# Abstract Builder
class Builder(object):
    def __init__(self):
        self.building = None

    def new_building(self):
        self.building = Building()

'''基于类的多态特性，重构builder'''
# Concrete Builder
class BuilderHouse(Builder):
    def build_floor(self):
        self.building.floor = 'One'

    def build_size(self):
        self.building.size = 'Big'

class BuilderFlat(Builder):
    def build_floor(self):
        self.building.floor = 'More than One'

    def build_size(self):
        self.building.size = 'Small'

# 构建结果
# Product
class Building(object):
    def __init__(self):
        self.floor = None
        self.size = None

    def __repr__(self):
        return 'Floor: %s | Size: %s' % (self.floor, self.size)


# 实例方法运行
# director = Director()
# director.builder = BuilderHouse()
# director.construct_building()
# building = director.get_building()
# print(building)
#
# director.builder = BuilderFlat()
# director.construct_building()
# building = director.get_building()
# print(building)


'''
4. Prototype（原型）
意图：
用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象。

适用性：
当要实例化的类是在运行时刻指定时，例如，通过动态装载；或者为了避免创建一个与产品类层次平行的工厂类层次时；
或者当一个类的实例只能有几个不同状态组合中的一种时。
建立相应数目的原型并克隆它们可能比每次用合适的状态手工实例化该类更方便一些。
'''

import copy

# 可用于构建存储数据类
class Prototype:
    def __init__(self):
        self._objects = {}

    def register_object(self, name, obj):
        """Register an object"""
        self._objects[name] = obj

    def unregister_object(self, name):
        """Unregister an object"""
        del self._objects[name]

    def clone(self, name, **attr):
        """Clone a registered object and update inner attributes dictionary"""
        obj = copy.deepcopy(self._objects.get(name))
        obj.__dict__.update(attr)
        return obj

def main():
    class A:
        def __str__(self):
            return "I am A"

    a = A()
    prototype = Prototype()
    prototype.register_object('a', a)
    b = prototype.clone('a', a=1, b=2, c=3)

    print(a)
    print(b.a,b.b,b.c)

# 示例运行
# main()


'''
5. Singleton（单例）
意图：
保证一个类仅有一个实例，并提供一个访问它的全局访问点。

适用性：
当类只能有一个实例而且客户可以从一个众所周知的访问点访问它时。
当这个唯一实例应该是通过子类化可扩展的，并且客户应该无需更改代码就能使用一个扩展的实例时。
'''


class Singleton(object):
    ''''' A python style singleton '''

    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'): #hasattr 判断cls对象是否包含属性_instance
            org = super(Singleton, cls) #super 函数是用于调用父类(超类)的一个方法
            cls._instance = org.__new__(cls, *args, **kw)
        return cls._instance


class SingleSpam(Singleton):

    def __init__(self, s):
        self.s = s

    def __str__(self):
        return self.s

# s1 = SingleSpam('spam')
# print (id(s1), s1)
# s2 = SingleSpam('spa')
# print (id(s2), s2)
# print (id(s1), s1)





