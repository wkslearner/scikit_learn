
''''''

# 基础只是点
'''
1、self表示一个具体的实例本身。如果用了staticmethod，那么就可以无视这个self，将这个方法当成一个普通的函数使用。
2、cls表示这个类本身。
'''


class A():
    def foo1(self):
        print ("Hello", self)

    @staticmethod  #转换成静态方法
    def foo2():
        print ("hello")

    @classmethod   # 转换成类方法
    def foo3(cls):
        print ("hello", cls.foo2())

# a=A()
# print(a.foo1())
# print(a.foo2())
# print(a.foo3())


'''
使用类进行数据格式定义
'''

_formats = {
    'ymd':'{d.year}-{d.month}-{d.day}',
    'mdy':'{d.month}/{d.day}/{d.year}',
    'dmy':'{d.day}/{d.month}/{d.year}'
    }

class Date:

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)



'''
类中函数属性化
https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p06_create_managed_attributes.html
'''

import math
class Circle:
    def __init__(self, radius):
        self.radius = radius

    #这里property  实现函数属性化访问  即可以以属性的方式调用函数
    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def diameter(self):
        return self.radius * 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

# a=Circle(5)
# print(a.area())



'''
类中描述器类的使用
一个描述器就是一个实现了三个核心的属性访问操作(get, set, delete)的类
它可以实现很多高级功能，并且它也是很多高级库和框架中的重要工具之一。
描述器通常是那些使用到装饰器或元类的大型框架中的一个组件
当程序中有很多重复代码的时候描述器就很有用
'''


class Integer:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Expected an int')
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class Point:
    # 这里定义传入参数必须是整型数据
    x = Integer('x')
    y = Integer('y')

    def __init__(self, x, y):
        self.x = x
        self.y = y


# 描述器类结合装饰器使用
# Descriptor for a type-checked attribute
class Typed: # 描述器类
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

# Class decorator that applies it to selected attributes
def typeassert(**kwargs):  # 装饰器
    # print(kwargs)
    def decorate(cls):
        for name, expected_type in kwargs.items():
            # Attach a Typed descriptor to the class
            setattr(cls, name, Typed(name, expected_type))
        return cls
    return decorate

# 使用装饰器装饰类
# Example use
@typeassert(name=str, shares=int, price=float)
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

# stc=Stock('xs',10,20.5)
# print(stc.name,stc.shares,stc.price)


'''
简化数据结构的初始化
通过类简化 新类属性过多的问题
'''

import math
class Structure1:
    # Class variable that specifies expected fields
    _fields = []
    # 这个类可以接受任意长度属性
    def __init__(self, *args):
        print(self._fields)
        print(args)
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        # Set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)


class Circle(Structure1):
    _fields = ['radius']

    def area(self):
        return math.pi * self.radius ** 2



'''
数据类型和赋值验证框架
多层类的继承
'''
# Base class. Uses a descriptor to set a value
# 描述器类
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):

        instance.__dict__[self.name] = value


# Descriptor for enforcing types
# 检验数据类型类 __init__ 函数放到继承的类里实现
class Typed(Descriptor):
    expected_type = type(None)

    def __set__(self, instance, value):

        if not isinstance(value, self.expected_type):
            raise TypeError('expected ' + str(self.expected_type))
        super().__set__(instance, value)


# Descriptor for enforcing values
# 检验值是否满足条件
class Unsigned(Descriptor):
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super().__set__(instance, value)


class MaxSized(Descriptor):
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super().__set__(instance, value)


class Integer(Typed):
    expected_type = int

# print(Integer(10).__set__({'price':15},10))
# 使用多继承实现多个功能
class UnsignedInteger(Integer, Unsigned):
    pass

class Float(Typed):
    expected_type = float

class UnsignedFloat(Float, Unsigned):
    pass

class String(Typed):
    expected_type = str

class SizedString(String, MaxSized):
    pass

class Stock:
    # Specify constraints  使用类实现值输入检测与限制
    name = SizedString('name', size=8)
    shares = UnsignedInteger('shares')
    price = UnsignedFloat('price')

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

print(Stock('XX',10,21.3))


'''
自定义容器
'''
import collections
class Items(collections.MutableSequence):
    def __init__(self, initial=None):
        self._items = list(initial) if initial is not None else []

    # Required sequence methods
    def __getitem__(self, index):
        print('Getting:', index)
        return self._items[index]

    def __setitem__(self, index, value):
        print('Setting:', index, value)
        self._items[index] = value

    def __delitem__(self, index):
        print('Deleting:', index)
        del self._items[index]

    def insert(self, index, value):
        print('Inserting:', index, value)
        self._items.insert(index, value)

    def __len__(self):
        print('Len')
        return len(self._items)



'''
在类中定义多个构造器
'''
import time
class Date:
    """方法一：使用类方法"""
    # Primary constructor  主要构造器
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    # Alternate constructor  代理构造器
    @classmethod  # 把方法转变成类方法
    def today(cls):
        t = time.localtime()
        return cls(t.tm_year, t.tm_mon, t.tm_mday)



'''
混入类的使用
'''
class LoggedMappingMixin:
    """
    Add logging to get/set/delete operations for debugging.
    """
    __slots__ = ()  # 混入类都没有实例变量，因为直接实例化混入类没有任何意义

    def __getitem__(self, key):
        print('Getting ' + str(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        print('Setting {} = {!r}'.format(key, value))
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        print('Deleting ' + str(key))
        return super().__delitem__(key)



class SetOnceMappingMixin:
    '''
    Only allow a key to be set once.
    '''
    __slots__ = ()

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(str(key) + ' already set')
        return super().__setitem__(key, value)


class StringKeysMappingMixin:
    '''
    Restrict keys to strings only
    '''
    __slots__ = ()

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('keys must be strings')
        return super().__setitem__(key, value)

class LoggedDict(LoggedMappingMixin, dict):
    pass

# d = LoggedDict()
# d['x'] = 23  # 调用__setitem__ 方法
# print(d['x']) # 调用__getitem__ 方法
# del d['x']  # 调用 __delitem__ 方法

'''
装饰器与类的结合使用
'''
def LoggedMapping(cls):
    """第二种方式：使用类装饰器"""
    cls_getitem = cls.__getitem__
    cls_setitem = cls.__setitem__
    cls_delitem = cls.__delitem__

    def __getitem__(self, key):
        print('Getting ' + str(key))
        return cls_getitem(self, key)

    def __setitem__(self, key, value):
        print('Setting {} = {!r}'.format(key, value))
        return cls_setitem(self, key, value)

    def __delitem__(self, key):
        print('Deleting ' + str(key))
        return cls_delitem(self, key)

    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__
    cls.__delitem__ = __delitem__
    return cls

@LoggedMapping
class LoggedDict(dict):
    pass



'''
创建缓存实例
'''

# The class in question
class Spam:
    def __init__(self, name):
        self.name = name

# Caching support
import weakref
_spam_cache = weakref.WeakValueDictionary()
def get_spam(name):
    print(_spam_cache.values())
    if name not in _spam_cache:
        s = Spam(name)
        _spam_cache[name] = s
    else:
        s = _spam_cache[name]
    return s


# a = get_spam('foo')
# b = get_spam('bar')
# c = get_spam('foo')

'''
使用元类控制实例的创建
'''

class NoInstances(type):
    def __call__(self, *args, **kwargs):
        raise TypeError("Can't instantiate directly")

# Example
class Spam(metaclass=NoInstances):  # 限定类实例的创建
    @staticmethod
    def grok(x):
        print('Spam.grok')

# ss=Spam()   # 报错
# s=Spam.grok(20)  # 正常


'''
抽象类
不可实例化，继承的子类才可实例化
'''

from abc import ABC, abstractmethod

class Foo(ABC):
    @abstractmethod
    def fun(self):
        '''please Implemente in subclass'''


class SubFoo(Foo):
    def fun(self):
        print('fun in SubFoo')

a = Foo()
a.fun()





