# PySaprk 将 DataFrame 数据保存为 Hive 分区表

创建 SparkSession
```
from pyspark.sql import SparkSession

spark = SparkSession.builder.enableHiveSupport().appName('test_app').getOrCreate()
sc = spark.sparkContext
hc = HiveContext(sc)
```

## 1. Spark创建分区表

```
# 可以将append改为overwrite，这样如果表已存在会删掉之前的表，新建表
df.write.saveAsTable(save_table, mode='append', partitionBy=['pt_day'])
```
saveAsTable 会自动创建hive表，partitionBy指定分区字段，默认存储为 parquet 文件格式。对于从文件生成的DataFrame，字段类型也是自动转换的，有时会转换成不符合要求的类型。

需要自定义字段类型的，可以在创建DataFrame时指定类型：
```
from pyspark.sql.types import StringType, StructType, BooleanType, StructField

schema = StructType([
    StructField("vin", StringType(), True),
    StructField("cust_id", StringType(), True),
    StructField("is_maintain", BooleanType(), True),
    StructField("is_wash", BooleanType(), True),
    StructField("pt_day", StringType(), True),
]
)

data = pd.read_csv('/path/to/data.csv', header=0)
df = spark.createDataFrame(data, schema=schema)
# 写入hive表时就是指定的数据类型了
df.write.saveAsTable(save_table, mode='append', partitionBy=['pt_day'])
```

## 2、向已存在的表插入数据
### 2.1 Spark创建的分区表

这种情况其实和建表语句一样就可以了
不需要开启动态分区
```
df.write.saveAsTable(save_table, mode='append', partitionBy=['pt_day'])
```

### 2.2 在Hive命令行或者Hive sql语句创建的表

- 这里主要指和Spark创建的表的文件格式不一样，Spark默认的文件格式为PARQUET，为在命令行Hive默认的文件格式为TEXTFILE，这种区别，也导致了异常的出现。
```
pyspark.sql.utils.AnalysisException: u"The format of the existing table default.csd_test_partition is `HiveFileFormat`. It doesn't match the specified format `ParquetFileFormat`.;"
```
- 需要开启动态分区, 不开启会有异常：
```
org.apache.spark.SparkException: Dynamic partition strict mode requires at least one static partition column. To turn this off set hive.exec.dynamic.partition.mode=nonstrict
```
代码：
```
# 建表
sql_str = "CREATE TABLE IF NOT EXISTS default.csd_test_partition (cust_id string, vin string, is_maintain boolean, is_wash boolean) partitioned by (pt_day string)"
hc.sql(sql_str) 
# 开启动态分区
spark.sql("set hive.exec.dynamic.partition.mode = nonstrict")
spark.sql("set hive.exec.dynamic.partition=true")
# 指定文件格式
df.write.saveAsTable(save_table, format='Hive', mode='append', partitionBy=['pt_day'])
```

通过临时视图创建

```
# 数据包含分区字段的，不要指定分区，要开启动态分区
df1.createTempView('temp_table')
# 或 df1.registerTempTable('temp_table')
hc.sql('insert into default.csd_test_partition select * from temp_table')
```

```
# 数据不包含分区字段的，可以直接指定分区插入，可以不用开启动态分区
df2 = df1.drop('pt_day')
df2.registerTempTable('temp_table')
hc.sql('insert into default.csd_test_partition partition(pt_day="20190516") select * from temp_table1')
```

## 3、总结

### 3.1 df.write.saveAsTable() 方法

mode='overwrite' 模式时，会创建新的表，若表名已存在则会被删除，整个表被重写。而 mode='append' 模式会在直接在原有数据增加新数据。

### 3.2 sql 语句进行插入

sql 语句插入只能先行建表，在执行插入操作。

`INSERT INTO tableName PARTITION(pt=pt_value) select * from temp_table` 的语句类似于 append 追加的方式。

`INSERT OVERWRITE TABLE tableName PARTITION(pt=pt_value) SELECT * FROM temp_table` 的语句能指定分区进行重写，而不会重写整张表。

sql 语句的方式比 `.write.saveAsTable()` 方法更灵活。


### 3.2 保存 hive 表的文件数量设置

默认的方式将会在hive分区表中保存大量的小文件，在保存之前对 DataFrame 用 .repartition() 重新分区，这样就能控制保存的文件数量。

如：
```
df.repartition(5).write.saveAsTable(...)
或
df.repartition(5).registerTempTable('temp_table')
```

以上设置一个分区只会保存 5 个数据文件。