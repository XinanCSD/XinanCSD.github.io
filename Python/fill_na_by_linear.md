# 数据预处理：PySpark 的实现线性插值填充缺失值


## 1. Python 实现线性插值填充缺失值

实现函数为：
```text
def linear_insert(x1, y1, x2, y2, insert_x):
    if type(insert_x) == int:
        insert_x = [insert_x]
    k = (y2 - y1) / (x2 - x1)
    return [k * (x - x1) + y1 for x in insert_x]


def fill_na_by_linear(lst):
    first_flag = False
    first_na = 0
    length = len(lst)
    for i in range(length):
        item = lst[i]
        if not first_flag:
            if item is None:
                first_na = i
                if first_na == 0:
                    # 第一个缺失值填充为 0
                    lst[0] = 0.0
                    continue
                first_flag = True
        else:
            if item is not None:
                first_flag = False
                lst[first_na:i] = linear_insert(first_na - 1, lst[first_na - 1], i, lst[i], range(first_na, i))

    if first_flag:
        lst[first_na:] = linear_insert(first_na - 2, lst[first_na - 2], first_na - 1, lst[first_na - 1], range(first_na, length))

    return lst
```

创建具有缺失值的序列：
```text
data = list(range(20))
data[0:3] = [None] * 3
data[10:16] = [None] * 6
data[-1] = None
print(data)
```
输出结果如下：
```text
[None, None, None, 3, 4, 5, 6, 7, 8, 9, None, None, None, None, None, None, 16, 17, 18, None]
```

填充缺失值：
```text
data_fill_na = fill_na_by_linear(data)
print(data_fill_na)
```
输出结果：
```text
[0.0, 1.0, 2.0, 3, 4, 5, 6, 7, 8, 9, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16, 17, 18, 19.0]
```


## 2. PySpark 的实现

创建 DataFrame:
```text
import pyspark

spark = pyspark.sql.SparkSession.builder.getOrCreate()
data = list(map(float, range(20)))
data1 = data.copy()
data[0:3] = [None] * 3
data[10:16] = [None] * 6
data[-1] = None
df = spark.createDataFrame([[1, i, j] for i, j in zip(data1, data)], schema=['col1', 'col2', 'col3'])
df.show()
```
输出结果
```text
+----+----+----+
|col1|col2|col3|
+----+----+----+
|   1| 0.0|null|
|   1| 1.0|null|
|   1| 2.0|null|
|   1| 3.0| 3.0|
|   1| 4.0| 4.0|
|   1| 5.0| 5.0|
|   1| 6.0| 6.0|
|   1| 7.0| 7.0|
|   1| 8.0| 8.0|
|   1| 9.0| 9.0|
|   1|10.0|null|
|   1|11.0|null|
|   1|12.0|null|
|   1|13.0|null|
|   1|14.0|null|
|   1|15.0|null|
|   1|16.0|16.0|
|   1|17.0|17.0|
|   1|18.0|18.0|
|   1|19.0|null|
+----+----+----+
```

将 DataFrame 转为 RDD 可以实现按行操作，需要有一列分区窗口，代码如下：
```text
def fill_na_by_columns(df: pyspark.sql.DataFrame, fill_col):
    schema = df.schema
    columns = df.columns
    fill_idx = columns.index(fill_col)

    def fill(x):
        out = []
        lst = []
        for v in x:
            r = [i for i in v]
            lst.append(r[fill_idx])
            out.append(r)

        lst = fill_na_by_linear(lst)
        for i in range(len(lst)):
            out[i][fill_idx] = lst[i]
        return out

    def get_key(item):
        return item.col2

    rdd = df.rdd.groupBy(lambda x: x.col1).mapValues(list)
    rdd = rdd.mapValues(lambda x: sorted(x, key=get_key))   # 对 col2 的排序可以取消
    rdd = rdd.mapValues(fill).flatMapValues(lambda x: x)
    rdd = rdd.map(lambda v: v[1])
    df_out = spark.createDataFrame(rdd, schema=schema)
    return df_out


data = fill_na_by_columns(df, 'col3')
data.show()
```
最后得到填充缺失值后的结果：
```text
+----+----+----+
|col1|col2|col3|
+----+----+----+
|   1| 0.0| 0.0|
|   1| 1.0| 1.0|
|   1| 2.0| 2.0|
|   1| 3.0| 3.0|
|   1| 4.0| 4.0|
|   1| 5.0| 5.0|
|   1| 6.0| 6.0|
|   1| 7.0| 7.0|
|   1| 8.0| 8.0|
|   1| 9.0| 9.0|
|   1|10.0|10.0|
|   1|11.0|11.0|
|   1|12.0|12.0|
|   1|13.0|13.0|
|   1|14.0|14.0|
|   1|15.0|15.0|
|   1|16.0|16.0|
|   1|17.0|17.0|
|   1|18.0|18.0|
|   1|19.0|19.0|
+----+----+----+
```