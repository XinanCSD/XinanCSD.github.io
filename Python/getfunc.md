# Python 通过函数名的字符串调用对应的函数

使用 getattr() 函数，可以获得类实例对象的属性方法，从而可以通过字符串调用函数：

```python
class func_factory:
    def func_name1(self, x):
        print('func_name1', x)
    def func_name2(self, x):
        print('func_name2', x)

f_factory = func_factory()
methodCall = getattr(f_factory, 'func_name1')
methodCall(1)
# 输出：func_name1 1
methodCall = getattr(f_factory, 'func_name2')
methodCall(2)
# 输出：func_name2 2
```