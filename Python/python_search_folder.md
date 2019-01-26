# Python 读取指定目录及其子目录下所有文件名

#### 目标
> 磁盘中的文件随着积累越来越多，当要寻找某个文件时，使用 windows 的搜索速度太慢且占内存。因此想要寻找一个可以获取指定目录下的所有文件的文件名，作为一种图书馆索引目录式的文件管理方式。

在此使用 python 的`os.walk()` 函数实现遍历指定目录及所有子目录下的所有文件。使用 python 3.6 版本实现。
walk()函数返回目录树生成器(迭代器)。通过自顶向下遍历目录来生成目录树中的文件名。对于根目录顶部（包括顶部本身）树中的每个目录，它产生一个3元组（dirpath，dirnames，filenames）。dirpath是一个字符串，即目录的路径。

dirnames是dirpath中子目录的名称列表。filenames是dirpath中非目录文件名称的列表。但列表中的名称不包含路径，要得到一个完整路径（从顶部开始）到dirpath中的文件或目录，请执行`os.path.join（dirpath，name）`。更多详情可查看 python 标准库文档[os.walk()](https://docs.python.org/3/library/os.html#os.walk) 。

实现代码如下

```python
import os
def all_path(dirname):
    filelistlog = dirname + "\\filelistlog.txt"  # 保存文件路径
    postfix = set(['pdf','doc','docx','epub','txt','xlsx','djvu','chm','ppt','pptx'])  # 设置要保存的文件格式
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if True:        # 保存全部文件名。若要保留指定文件格式的文件名则注释该句
            #if apath.split('.')[-1] in postfix:   # 匹配后缀，只保存所选的文件格式。若要保存全部文件，则注释该句
                try:
                    with open(filelistlog, 'a+') as fo:
                        fo.writelines(apath)
                        fo.write('\n')
                except:
                    pass    # 所有异常全部忽略即可

    
if __name__ == '__main__':
    dirpath = "D:"  # 指定根目录
    all_path(dirpath)
```


程序运行结束将所有文件名保存为指定目录下的 filelistlog.txt 文件。
