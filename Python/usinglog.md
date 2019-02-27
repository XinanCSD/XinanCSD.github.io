# python 的 logging 模块日志功能使用详解

##  目录

-   [一、logging 基本用法](#一logging-基本用法)
    -   [1、添加日志记录](#添加日志记录)
    -   [2、日志调用函数的使用](#日志调用函数的使用)
    -   [3、日志输出级别](#日志输出级别)
    -   [4、设置日志输出格式](#设置日志输出格式)
-   [二、创建 logger 对象](#二创建-logger-对象)
-   [三、给库或者模块添加日志记录](#三给库或者模块添加日志记录)
-   [四、使用配置文件设置日志的配置信息](#四使用配置文件设置日志的配置信息)
-   [参考资料](#参考资料)

## 一、logging 基本用法

### 1、添加日志记录

给简单的程序添加日志功能，最简单的方法是使用 logging 模块，示例如下：


```python
# 导入 logging 模块
import logging

# 配置 logging 系统
logging.basicConfig(level=logging.DEBUG)

# 添加 log 记录示例
logging.critical('logging critical message.')
logging.error('logging error message')
logging.warning('logging warning message')
logging.info('logging info message')
logging.debug('logging debug message')
```

运行这个程序，会在控制台输入这样的日志信息：
```text
CRITICAL:root:logging critical message.
ERROR:root:logging error message
WARNING:root:logging warning message
INFO:root:logging info message
DEBUG:root:logging debug message
```

### 2、日志调用函数的使用

每个日志操作 (critical(), error(), warning(), info(), debug()) 的参数都是一条字符串消息，当产生日志消息时，可以使用 % 操作符提供的参数格式化字符串消息。如：
```text
msg = 'foo'
i = 1
logging.info('info message: %s', msg)
logging.info('this is a number: %d', i)
```
输出：
```text
INFO:root:info message: foo
INFO:root:this is a number: 1
```

对于 error() 函数，可以传入 `exc_info=True` 参数设置追踪错误信息：
```text
try:
    int('a')
except ValueError:
    logging.error('type error', exc_info=True)
```
输出为：
```text
ERROR:root:type error
Traceback (most recent call last):
  File ".../test.py", line 19, in <module>
    int('a')
ValueError: invalid literal for int() with base 10: 'a'
```

### 3、日志输出级别

这 5 个 logging 调用 (critical(), error(), warning(), info(), debug()) 分别代表不同的严重级别，以降序排列。可以通过设置日志输出级别来选择需要输出的日志信息，Logging 的输出级别如下表所示：

| Level | Numeric value |
|----|----|
| CRITICAL|	50 |
| ERROR |	40 |
| WARNING |	30 |
| INFO |	20 |
| DEBUG	| 10 |
| NOTSET |	0 |

basicConfig() 的 level 参数是一个过滤器，所有等级低于此设定的消息都会被忽略掉，如上面的设置 level=logging.DEBUG，则会输出所有记录。而当输出级别设置为 ERROR 时，level=logging.ERROR，只会输出 critical(), error() 的信息：
```text
CRITICAL:root:logging critical message.
ERROR:root:logging error message
```

### 4、设置日志输出格式

可以通过修改调用 basicConfig() 的参数设置日志输出的格式。

- `format` 参数设置输出日志记录的格式，参数的具体格式设置参考 [LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes)。
- `datefmt` 设置输出时间的格式，参数格式接受 [time.strftime()](https://docs.python.org/3/library/time.html#time.strftime) 的格式。

参数设置示例如下：
```text
logging.basicConfig(
    level=logging.DEBUG,
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

输出结果变成了下面这样：
```text
2019/01/26 21:16:46 - root - CRITICAL - logging critical message.
2019/01/26 21:16:46 - root - ERROR - logging error message
2019/01/26 21:16:46 - root - WARNING - logging warning message
2019/01/26 21:16:46 - root - INFO - logging info message
2019/01/26 21:16:46 - root - DEBUG - logging debug message
```

- `filename` 指定输出的日志文件，示例：
```text
logging.basicConfig(
    filename='example.log',
    level=logging.DEBUG,
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```
则会生成 `example.log` 日志文件输出相应的日志记录。

- `stream` 使用一个特殊的流对象初始化 [StreamHandler](https://docs.python.org/3/library/logging.handlers.html#logging.StreamHandler)
```text
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```
会将日志信息输出到系统标准输出。

注意：`filename` 和 `stream` 不能同时指定，否则会抛出 ValueError 异常。若需要设定多种输出方式，可以通过 `handlers` 参数传入一个 handler 列表：
```text
from logging import StreamHandler, FileHandler
import sys

sh = StreamHandler(sys.stdout)
fh = FileHandler('example.log')

# 配置 logging 系统
logging.basicConfig(
    handlers=[sh, fh],
    level=logging.DEBUG,
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```
通过这样的设置，会同时在系统输出和 `example.log` 日志文件种输出日志信息。


## 二、创建 logger 对象

可以通过 [logging.getLogger(name=None)](https://docs.python.org/3/library/logging.html#logging.getLogger) 创建一个指定名称的 Logger 对象（logging 系统默认的 logger 名称为 `root`）。

```python
import logging
import sys

logger = logging.getLogger('testLogger')
logger.setLevel(logging.DEBUG)
# 消息格式化
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

# 日志文件输出
fh = logging.FileHandler('example.log')
fh.setFormatter(formatter)
logger.addHandler(fh)   # 给logger对象添加 handler

# 系统输出
# StreamHandler
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

logger.critical('logger critical message.')
logger.error('logger error message')
logger.warning('logger warning message')
logger.info('logger info message')
logger.debug('logger debug message')
```

输出的日志信息变成：
```text
2019/01/26 21:49:49 - testLogger - CRITICAL - logger critical message.
2019/01/26 21:49:49 - testLogger - ERROR - logger error message
2019/01/26 21:49:49 - testLogger - WARNING - logger warning message
2019/01/26 21:49:49 - testLogger - INFO - logger info message
2019/01/26 21:49:49 - testLogger - DEBUG - logger debug message
```
与第一节中的区别仅在于 logger 的名称，其他设置的效果相同。


## 三、给库或者模块添加日志记录

想给一个库添加日志功能，但又不希望它影响到那些没有使用日志功能的程序。由于使用日志的环境是未知的，因而不能在库代码中尝试去自行配置日志系统，或者对已有的日志配置做任何假设。

对于想执行日志记录的库来说，应该创建一个专用的日志对象并将其初始化为 NullHandler，在单独的 testlib.py 模块中输入下面的函数。
```python
# testlib.py

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def foo():
    logger.info('info: this is foo message')
    print('print: foo')
```

`getLogger(__name__)` 创建一个与模块同名的一个专有的日志对象，由于所有模块是唯一的，这样就于其他日志对象隔开了。

`logger.addHandler(logging.NullHandler())` 操作绑定一个空的处理例程到刚才的日志对象。默认情况下，空 handler 会忽略所有日志消息。因此，用到这个库且日志系统从未配置过，那么就不会出现任何日志信息或警告信息。

默认情况下将不会产生任何日志输出。例如：
```text
import testlib
testlib.foo()
```
仅输出 print() 函数的内容：
```text
print: foo
```
但是如果日志系统得到适当的配置，则日志消息将开始出现。
```text
import logging
import testlib
logging.basicConfig(level=logging.DEBUG)

testlib.foo()
```
此时输出为：
```text
print: foo
INFO:testlib:info: this is foo message
```
同时，在调用模块的程序中也能定义自身的 logger，示例如下：
```text
# testmian.py
import logging
import testlib
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('testLogger')

testlib.foo()
logger.info('this is test logger info')
```
所有的日志信息都会输出：
```text
INFO:testlib:info: this is foo message
print: foo
INFO:testLogger:this is test logger info
```

## 四、使用配置文件设置日志的配置信息

上面的日志配置都被直接硬编码到了程序中，这么做不是好的编程规范，`logging.config`，提供了 `logging.config.fileConfig()` 方法可以从配置文件中进行日志配置。

把 `logging.basicConfig()` 调用修改程如下形式：
```python
import logging
import logging.config


logging.config.fileConfig('logconfig.ini')

logging.critical('logger critical message.')
logging.error('logger error message')
logging.warning('logger warning message')
logging.info('logger info message')
logging.debug('logger debug message')
```

然后创建一个配置文件 `logconfig.ini`，配置文件的格式具有如下的形式：
```text
[loggers]
keys=root

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(name)s - %(levelname)s - %(message)s
```
`[loggers]` 配置项指定需要配置的logger，`root` 是logging系统默认logger。若有多个值，用逗号分隔`keys=root, testLogger`。然后在`[logger_%(name)s]` 配置项配置各个logger的具体配置信息。其他日志对象的设置需要如下设置：
```text
[logger_testLogger]
level=INFO
handlers=consoleHandler
qualname=testLogger
```
对于没有单独设定的日志对象，将使用默认的 `root` 系统设置。

将输出设置为日志文件：
```text
[loggers]
keys=root

[handlers]
keys=fileHandler, errorFileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=fileHandler, errorFileHandler

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('example.log', 'a')

[handler_errorFileHandler]
class=FileHandler
level=ERROR
formatter=simpleFormatter
args=('example-error.log', 'a')

[formatter_simpleFormatter]
format=%(name)s - %(levelname)s - %(message)s
```
上面的配置同时还将错误信息单独输出到错误日志中。更过关于文件的配置可以参考 [logging.config.fileConfig()](https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig) 文档。

---
## 参考资料

- 《Python Cookbook 第3版》中文版
- python 标准库文档： [logging](https://docs.python.org/3/library/logging.html), [logging.config](https://docs.python.org/3/library/logging.config.html), [logging.handlers](https://docs.python.org/3/library/logging.handlers.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html)