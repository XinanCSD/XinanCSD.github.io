# pyspark 实现对列累积求和


pandas 的 `cumsum()` 函数可以实现对列的累积求和。使用示例如下：

```
import pandas as pd
data = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
data = pd.DataFrame(data, columns=['diff'])
data['cumsum_num'] = data['diff'].cumsum()
print(data)
```
输出结果：
```
    diff  cumsum_num
0      1           1
1      0           1
2      0           1
3      0           1
4      1           2
5      0           2
6      0           2
7      1           3
8      0           3
9      0           3
10     0           3

Process finished with exit code 0
```


对于 pyspark 没有 `cumsum()` 函数可以直接进行累加求和，若要实现累积求和可以通过对一列有序的列建立排序的 Window 进行求和，代码如下所示：

创建 DataFrame 对象：
```python
import pyspark
from pyspark.sql import functions as fn
from pyspark.sql import SparkSession
from pyspark.sql import Window
import pandas as pd

conf = pyspark.SparkConf().setAll([])
spark_session = SparkSession.builder.appName('test_app').config(conf=conf).getOrCreate()
sc = spark_session.sparkContext
sc.setLogLevel('WARN')

data = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
data = pd.DataFrame(data, columns=['diff'])
data['number'] = range(len(data))
data = spark_session.createDataFrame(data, schema=['diff', 'number'])
data.show()
```
原 DataFrame 数据：
```text
+----+------+
|diff|number|
+----+------+
|   1|     0|
|   0|     1|
|   0|     2|
|   0|     3|
|   1|     4|
|   0|     5|
|   0|     6|
|   1|     7|
|   0|     8|
|   0|     9|
|   0|    10|
+----+------+
```

根据 number 排序实现累积求和：
```text
win = Window.orderBy('number')
data = data.withColumn('cumsum_num', fn.sum(data['diff']).over(win))
data.show()
```
结果为：
```text
+----+------+----------+
|diff|number|cumsum_num|
+----+------+----------+
|   1|     0|         1|
|   0|     1|         1|
|   0|     2|         1|
|   0|     3|         1|
|   1|     4|         2|
|   0|     5|         2|
|   0|     6|         2|
|   1|     7|         3|
|   0|     8|         3|
|   0|     9|         3|
|   0|    10|         3|
+----+------+----------+
```