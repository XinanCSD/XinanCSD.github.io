# spark-submit 命令使用详解

spark-submit 用户打包 Spark 应用程序并部署到 Spark 支持的集群管理气上，命令语法如下：
```text
spark-submit [options] <python file> [app arguments]
```
app arguments 是传递给应用程序的参数，常用的命令行参数如下所示：
- --master: 设置主节点 URL 的参数。支持：
    - local： 本地机器。
    - spark://host:port：远程 Spark 单机集群。
    - yarn：yarn 集群
- --deploy-mode：允许选择是否在本地（使用 client 选项）启动 Spark 驱动程序，或者在集群内（使用 cluster 选项）的其中一台工作机器上启动。默认值是 client。
- --name：应用程序名称，也可在程序内设置。
- --py-files：.py, .egg 或者 .zip 文件的逗号分隔列表，包括 Python 应用程序。这些文件将分发给每个执行节点。
- --files：逗号分隔的文件列表，这些文件将分发给每个执行节点。
- --conf：动态地改变应用程序的配置。
- --driver-memory：指定应用程序在驱动节点上分配多少内存的参数，类似与 10000M， 2G。默认值是 1024M。
- --executor-memory：指定每个执行节点上为应用程序分配的内存，默认 1G。
- --num-executors：指定执行器节点数。
- --help：展示帮助信息和退出。

以下均是在 yarn 集群提交的任务。

1、默认设置: 会将所有日志和系统输出结果输出到 spark-submit 的 client 上
```
spark-submit --master yarn code1.py
```

code1.py
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Test_Code1').enableHiveSupport().getOrCreate()

spark.sql("select count(*) from default.test_table").show()
```

2、设置 Executor 的日志级别，Executor 执行的细节（WARN 以下级别的日志）不会输出到 client 中
```
spark-submit --master yarn code2.py
```

code2.py
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Test_Code1').enableHiveSupport().getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

spark.sql("select count(*) from default.test_table").show()
```

3、使用 cluster 模式

```
spark-submit --master yarn --deploy-mode cluster code1.py
```
--deploy-mode 可选 cluster 或 client，cluster 模式下，在 spark-submit 的 client 服务器上不会输出日志和系统输出，仅输出如下语句。只能在 Hadoop 集群上才能看到执行细节和输出
```text
2019-09-06 00:00:00 INFO  Client:54 - Application report for application_1556516318747_25363 (state: RUNNING)
```


4、自定义依赖的模块或读取文件
```
spark-submit --master yarn --files file1.txt --py-files code4.py code3.py 
```

code3.py
```python
from code4 import code4func
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Test_Code1').enableHiveSupport().getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

table = code4func()
with open("file1.txt", 'rt') as rf:
    db = rf.readline().strip()

spark.sql("select count(*) from {}.{}".format(db, table)).show()
```

code4.py
```python
def code4func():
    return "test_table"
```

file1.txt
```
default
```

自定义的 package 可以打包成 egg 文件上传(该部分代码参考 《PySpark 实战》P:178)。例如有一个自定义创建的 package：
```text
additionalCode/
├── setup.py
└── utilities
    ├── __init__.py
    ├── base.py
    ├── converters
    │   ├── __init__.py
    │   ├── base.py
    │   └── distance.py
    └── geoCalc.py
```
创建一个 egg 文件：
```text
python setup.py bdist_egg
```
生成了 dist 文件夹下的 `PySparkUtilities-0.1.dev0-py3.6.egg` 文件

提交作业：
```text
spark-submit --master yarn --py-files additionalCode/dist/PySparkUtilities-0.1.dev0-py3.6.egg calculatingGeoDistance.py
```

5、配置集群资源

当执行的 job 需要更多资源时，可以自定义配置使用的资源。
```text
spark-submit --master yarn --driver-memory 15g \
    --num-executors 10 --executor-cores 4 --executor-memory 15G \
    --conf "spark.executor.memoryOverhead=15G" \
    code1.py
```
或在程序内设置
```
spark-submit code5.py
```

code5.py
```python
import pyspark
from pyspark.sql import SparkSession

conf1 = pyspark.SparkConf().setAll([
            ('spark.executor.memory', '15g'),
            ('spark.executor.memoryOverhead', '16g'),
            ('spark.executor.cores', '4'),
            ('spark.num.executors', '10'),
            ('spark.driver.memory', '16g')])

spark = SparkSession.builder.appName('Test_Code1').enableHiveSupport().config(conf=conf1).getOrCreate()

spark.sql("select count(*) from default.test_table").show()
```

6、使用 Python 虚拟环境

当使用 cluster 或应用某些第三方包的时候，在 Executor 中会出现 ImportError 的错误，导致 job 执行失败，如下提交方式会报错：

```text
spark-submit --master yarn --deploy-mode cluster code6.py
```
报错信息：
```text
Traceback (most recent call last):
  File "code6.py", line 2, in <module>
    import numpy as np
ImportError: No module named numpy
```

这是由于节点中的 python 环境没有安装相应的依赖包，此时需要创建一个 python 虚拟环境并安装所有的依赖包。

创建虚拟环境 python-env，打包为 venv.zip：
```text
virtualenv python-env
```
venv.zip 部分目录结构如下所示：
```text
venv.zip
└──python-env/
    ├── bin
    │   └── python
    ├── include
    ├── lib
    └── lib64
```

spark-submit 命令：
```text
spark-submit --master yarn  --deploy-mode cluster \
    --archives ./venv.zip#env \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=env/python-env/bin/python \
    code6.py
```

code6.py
```python
from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.appName('Test_Code1').enableHiveSupport().getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

arr = np.array([1, 2, 3])
print(arr)

spark.sql("select count(*) from default.test_table").show()
```