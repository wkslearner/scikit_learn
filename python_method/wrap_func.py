''''''

'''
装饰器使用
'''


import time
from functools import wraps

#带参数装饰器
def timethis(prints):
    def print_if(func):
        @wraps(func)  #保留函数的基础信息
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if prints==True:
                print(func.__name__, end-start)
            return result
        return wrapper
    return print_if


@timethis(prints=True)
def countdown(n):
    while n > 0:
        n -= 1


'''
带参数装饰器
带参数过程是在已有装饰器外再加一层带参函数
'''

from functools import wraps, partial
import logging

def attach_wrapper(obj, func=None):
    if func is None:
        # print(partial(attach_wrapper, obj))
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def logged(level, name=None, message=None):

    def decorate(func):
        logname = name if name else func.__module__

        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__


        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)

        # Attach setter functions
        @attach_wrapper(wrapper)
        def set_level(newlevel):
            nonlocal level
            level = newlevel

        # 使用访问函数生成装饰器的属性,并改变原有参数值
        @attach_wrapper(wrapper)
        def set_message(newmsg):
            nonlocal logmsg
            logmsg = newmsg

        return wrapper

    return decorate


# Example use
@logged(logging.INFO,name='ss',message='xx')
def add(x, y):
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    print('haha Spam!')



'''
使用装饰器实现类型检查
'''
from inspect import signature
from functools import wraps

def typeassert(*ty_args, **ty_kwargs):
    def decorate(func):
        # If in optimized mode, disable type checking
        # 使用__debug__参数 调整是否使用装饰器
        if not __debug__:
            return func

        # Map function argument names to supplied types
        sig = signature(func)  # 返回函数的签名信息
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments  # 执行绑定
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)  # 执行绑定 但不允许忽略参数
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            'Argument {} must be {}'.format(name, bound_types[name])
                            )
            return func(*args, **kwargs)
        return wrapper
    return decorate


@typeassert(int, z=int)
def spam(x, y, z=42):
    print(x, y, z)


'''
装饰器与类的结合使用
'''

class A:
    # Decorator as an instance method
    def decorator1(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('Decorator 1')
            return func(*args, **kwargs)
        return wrapper

    # Decorator as a class method
    @classmethod
    def decorator2(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('Decorator 2')
            return func(*args, **kwargs)
        return wrapper


# As a class method
@A.decorator2
def grok():
    pass

'''
将装饰器定义为类
你想使用一个装饰器去包装函数，但是希望返回一个可调用的实例。 你需要让你的装饰器可以同时工作在类定义的内部和外部。
'''

import types
from functools import wraps

class Profiled:
    def __init__(self, func):
        wraps(func)(self)
        self.ncalls = 0

    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)


@Profiled
def add(x, y):
    return x + y

class Spam:
    @Profiled
    def bar(self, x):
        print(self, x)


'''
为包装函数增加参数
'''


from functools import wraps

def optional_debug(func):
    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)

    return wrapper

# 被包装函数多了一个参数
@optional_debug
def spam(a,b,c):
    print(a,b,c)



if __name__=='__main__':

    # countdown(100000)  # 使用装饰器
    # countdown.__wrapped__(100000)  #不使用装饰器

    # logging.basicConfig(level=logging.INFO)
    # add.set_message('lala')
    # add(1,2)

    # spam(1,'ss',3)

    # add(2,3)
    # print(add.ncalls)

    print(Spam().bar(1))




    pass







