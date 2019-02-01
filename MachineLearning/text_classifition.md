# python 实现中文文本分类

本文基于 Python 采用 scikit-learn 模块实现中文文本分类。

# 文本分类


## 一、预处理

### 1. 获取语料库

语料库数据选用搜狗语料库的搜狐新闻数据精简版：http://www.sogou.com/labs/resource/cs.php。

>数据集介绍：
<br>
来自搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据，提供URL和正文信息
<br>
格式说明：
数据格式为
```
<doc>

<url>页面URL</url>

<docno>页面ID</docno>

<contenttitle>页面标题</contenttitle>

<content>页面内容</content>

</doc>
```
注意：content字段去除了HTML标签，保存的是新闻正文文本

下载后解压到 SogouCS.reduced 文件夹。下载的文本是 xml 格式，需要解析为纯文本。参考这篇博文进行解析：http://www.sohu.com/a/147504203_609569。需要注意的是，下载的原文本数据中 缺少跟节点，并且有些特殊符号需要去掉，因此进行了一些格式处理步骤。代码如下所示，保存为 sougou_text.py：

```python
#!/usr/bin/python
# -*- encoding:utf-8 -*-
 
 
import os
from xml.dom import minidom
from urllib.parse import urlparse
import glob
from queue import Queue
from threading import Thread, Lock
import time

THREADLOCK = Lock()
# 解析的文本保存路径
corpus_dir = './SogouCS.corpus/'


def file_format(from_file, to_file):
    """对下载的文本进行格式处理"""
    try:
        # 原文本需要用 gb18030 打开
        with open(from_file, 'r', encoding='gb18030') as rf:
            lines = rf.readlines()
        # xml 格式有问题，需添加根节点
        lines.insert(0, '<data>\n')
        lines.append('</data>')
        with open(to_file, 'w', encoding='utf-8') as wf:
            for l in lines:
                l = l.replace('&', '')
                wf.write(l)
    except UnicodeDecodeError:
        print("转码出错",from_file)


def praser_handler(q: Queue):
    # 建立url和类别的映射词典
    dicurl = {'auto.sohu.com': 'qiche', 'it.sohu.com': 'hulianwang', 'health.sohu.com': 'jiankang',
              'sports.sohu.com': 'tiyu', 'travel.sohu.com': 'lvyou', 'learning.sohu.com': 'jiaoyu',
              'cul.sohu.com': 'wenhua', 'mil.news.sohu.com': 'junshi', 'business.sohu.com': 'shangye',
              'house.sohu.com': 'fangchan', 'yule.sohu.com': 'yule', 'women.sohu.com': 'shishang',
              'media.sohu.com': 'chuanmei', 'gongyi.sohu.com': 'gongyi', '2008.sohu.com': 'aoyun'}
    while not q.empty():
        file = q.get()
        with THREADLOCK:
            print("文件" + file)
        file_code = file.split('.')[-2]
        file_format(file, file) # 进行格式处理
        doc = minidom.parse(file)
        root = doc.documentElement
        claimtext = root.getElementsByTagName("content")
        claimurl = root.getElementsByTagName("url")
        textnum = len(claimurl)
        for index in range(textnum):
            if claimtext[index].firstChild is None:
                continue
            url = urlparse(claimurl[index].firstChild.data)
            if url.hostname in dicurl:
                if not os.path.exists(corpus_dir + dicurl[url.hostname]):
                    os.makedirs(corpus_dir + dicurl[url.hostname])
                fp_in = open(
                    corpus_dir + dicurl[url.hostname] + "/%s_%d.txt" % (file_code, index),"wb")
                fp_in.write((claimtext[index].firstChild.data).encode('utf8'))
                fp_in.close()


def sougou_text_praser(org_dir):
    # 用8个线程处理文本
    q = Queue()
    for file in glob.glob(org_dir + '*.txt'):
        q.put(file)
    for i in range(8):
        Thread(target=praser_handler, args=(q,)).start()
    while not q.empty():
        time.sleep(10)


def files_count(corpus_dir):
    # 统计各类别下的文本数
    folders = os.listdir(corpus_dir)
    total = 0
    for folder in folders:
        if folder.startswith('.DS'):
            continue
        fpath = os.path.join(corpus_dir, folder)
        files = os.listdir(fpath)
        num = len(files)
        total += num
        print(folder, num, sep=':')
    print('Total article:', total)


if __name__=="__main__":

    org_dir = './SogouCS.reduced/'
    sougou_text_praser(org_dir)
    files_count(corpus_dir)
```
直接运行脚本，解析后得到 13类 336960 篇文章。
```
shangye		60790
hulianwang	10736
fangchan	69661
jiaoyu		10045
qiche		6539
wenhua		3225
jiankang	5442
tiyu		83689
shishang	16412
aoyun		26437
lvyou		8760
yule		32335
junshi		2889
```

### 2. 数据集切分

将文本库切分为训练集和测试集，切分比例为 0.2，数据集切分后存储在SogouCS目录下train和test两个文件夹下。代码如下：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将文本分类数据集分为训练集和测试集
@author: CSD
"""
import glob
import os
import random
import shutil
from threading import Thread, Lock
from queue import Queue


THREADLOCK = Lock()


def check_dir_exist(dir):
    # 坚持目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)


def copyfile(q):
    while not q.empty():
        full_folder, train, test, divodd = q.get()
        files = glob.glob(full_folder)
        filenum = len(files)
        testnum = int(filenum * divodd)
        testls = random.sample(list(range(filenum)), testnum)
        for i in range(filenum):
            if i in testls:
                shutil.copy(files[i], os.path.join(test, os.path.basename(files[i])))
            else:
                shutil.copy(files[i], os.path.join(train, os.path.basename(files[i])))
        with THREADLOCK:
            print(full_folder)


def data_divi(from_dir, to_dir, divodd=0.2):
    train_folder = os.path.join(to_dir, "train")
    test_folder = os.path.join(to_dir, "test")
    check_dir_exist(train_folder)
    check_dir_exist(test_folder)

    q = Queue()

    for basefolder in os.listdir(from_dir):
        if basefolder.startswith('.DS'):
            continue
        full_folder = os.path.join(from_dir, basefolder)
        print(basefolder)
        train = os.path.join(train_folder, basefolder)
        check_dir_exist(train)
        test = os.path.join(test_folder,basefolder)
        check_dir_exist(test)
        full_folder += "/*.txt"
        q.put((full_folder, train, test, divodd))

    for i in range(8):
        Thread(target=copyfile, args=(q,)).start()


if __name__ == "__main__":
    corpus_dir = './SogouCS.corpus'
    exp_path = './SogouCS/'
    divodd = 0.2
    data_divi(corpus_dir, exp_path, divodd)
```


### 3. 中文分词

中文分词使用结巴分词器（详情请参考：[jieba](https://github.com/fxsjy/jieba/blob/master/README.md)），分析器加载自定义词典：
```text
jieba.load_userdict(userdict)
```
为了体现分词的过程，现将分词后的语料存储为train_seg和test_seg。
```python
import jieba
import os
from threading import Thread, Lock
from queue import Queue

userdict = './userdict.txt'
jieba.load_userdict(userdict)
LOCK = Lock()

def readfile(filepath, encoding='utf-8'):
    # 读取文本
    with open(filepath, "rt", encoding=encoding) as fp:
        content = fp.read()
    return content


def savefile(savepath, content):
    # 保存文本
    with open(savepath, "wt") as fp:
        fp.write(content)
        
def check_dir_exist(dir):
    # 坚持目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)

def text_segment(q):
    """
    对一个类别目录下进行分词
    """
    while not q.empty():
        from_dir, to_dir = q.get()
        with LOCK:
            print(from_dir)
        files = os.listdir(from_dir)
        for name in files:
            if name.startswith('.DS'):
                continue
            from_file = os.path.join(from_dir, name)
            to_file = os.path.join(to_dir, name)

            content = readfile(from_file)
            seg_content = jieba.cut(content)
            savefile(to_file, ' '.join(seg_content))


def corpus_seg(curpus_path, seg_path):
    """对文本库分词，保存分词后的文本库,目录下以文件归类 curpus_path/category/1.txt, 保存为 seg_path/category/1.txt"""
    check_dir_exist(seg_path)
    q = Queue()
    cat_folders = os.listdir(curpus_path)
    for folder in cat_folders:
        from_dir = os.path.join(curpus_path, folder)
        to_dir = os.path.join(seg_path, folder)
        check_dir_exist(to_dir)

        q.put((from_dir, to_dir))

    for i in range(len(cat_folders)):
        Thread(target=text_segment, args=(q,)).start()
if __name__ == '__main__':
    # 分词
    data_dir = './SogouCS/'
    corpus_seg(data_dir + 'train/', data_dir + 'train_seg')
    corpus_seg(data_dir + 'test/', data_dir + 'test_seg')
```


### 4. TFIDF 特征提取

用 tfidf 权重计算方法构建文档向量空间，用到 sklearn 特征提取模块的 [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) 类。构建词带对象，将特征矩阵保存为 sklearn 的 Bunch 数据结构 Bunch(filenames=[], label=[], tdm=[], vocabulary={})。其中 filenames 是文本绝对路径列表，label 是对应的标签，tdm 是特征矩阵，vocabulary 是语料库词典，为了统一向量空间，使用训练集的语料库词典。
代码如下，保存为 tfidf_feature.py：
```python
"""
文章分词
提供两种方案：
"""

import jieba
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from concurrent import futures
import sys
import pickle

userdict = '/Users/ivzbv/WorkLocal/programing/datasets/sogoudataset/userdict.txt'
jieba.load_userdict(userdict)


def tokenizer():
    return jieba
    
def readfile(filepath, encoding='utf-8'):
    # 读取文本
    with open(filepath, "rt", encoding=encoding) as fp:
        content = fp.read()
    return content


def savefile(savepath, content):
    # 保存文本
    with open(savepath, "wt") as fp:
        fp.write(content)


def writeobj(path, obj):
    # 持久化python对象
    with open(path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def readobj(path):
    # 载入python对象
    with open(path, "rb") as file_obj:
        obj = pickle.load(file_obj)
    return obj


def check_dir_exist(dir):
    # 坚持目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)


def folder_handler(args):
    """遍历一个文件夹下的文本"""
    folder, encoding, seg = args
    print('遍历：', folder)
    try:
        assert os.path.isdir(folder)
    except AssertionError:
        return None
    files = os.listdir(folder)
    content = []
    filenames = []
    for name in files:
        if name.startswith('.DS'):
            continue
        filepath = os.path.join(folder, name)
        text = readfile(filepath, encoding)
        # 在此可直接分词
        if seg:
            text = ' '.join(jieba.cut(text))
        content.append(text)
        filenames.append(filepath)
    return (filenames, content)


def corpus_bunch(data_dir, encoding='utf-8', seg=True, tier=2):
    """
    得到文本库，返回一个 Bunch 对象
    :param data_dir:    文本库目录，目录下以文件归类 data_dir/category/1.txt
    :param encoding:    文本库编码
    :param seg:         是否需要分词
    :param tier:        data_dir 目录下的层级 2: data_dir/category/1.txt, 1: data_dir/1.txt
    :return:
    """
    try:
        assert os.path.isdir(data_dir)
    except AssertionError:
        print('{} is not a folder!')
        sys.exit(0)
    try:
        assert tier in [1, 2]
    except AssertionError:
        print('目录层级 tier 只能是 1 或 2！')
        sys.exit(0)
    corpus = Bunch(filenames=[], label=[], contents=[])
    if tier == 2:
        folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if not d.startswith('.DS')]
    else:
        folders = [data_dir]
    # 创建线程池遍历二级目录
    with futures.ThreadPoolExecutor(max_workers=len(folders)) as executor:
        folders_executor = {executor.submit(folder_handler, (folder, encoding, seg)): folder for folder in folders}
        for fol_exe in futures.as_completed(folders_executor):
            folder = folders_executor[fol_exe]
            filenames, content = fol_exe.result()
            if content:
                cat_name = folder.split('/')[-1]
                content_num = len(content)
                print(cat_name, content_num, sep=': ')
                label = [cat_name] * content_num
                corpus.filenames.extend(filenames)
                corpus.label.extend(label)
                corpus.contents.extend(content)
    return corpus


def vector_space(corpus_dir, stop_words=None, vocabulary=None, encoding='utf-8', seg=True, tier=2):
    """将一个语料库向量化"""
    vectorizer = TfidfVectorizer(stop_words=stop_words, vocabulary=vocabulary)
    corpus = corpus_bunch(corpus_dir, encoding=encoding, seg=seg, tier=tier)
    tfidf_bunch = Bunch(filenames=corpus.filenames, label=corpus.label, tdm=[], vocabulary={})
    tfidf_bunch.tdm = vectorizer.fit_transform(corpus.contents)
    tfidf_bunch.vocabulary = vectorizer.vocabulary_
    return tfidf_bunch


def tfidf_space(data_dir, save_path, stopword_path=None, encoding='utf-8', seg=True):
    stpwd = None
    if stopword_path:
        stpwd = [wd.strip() for wd in readfile(stopword_path).splitlines()]
    check_dir_exist(save_path)
    train = data_dir + 'train'
    train_tfidf = vector_space(train, stop_words=stpwd, encoding=encoding, seg=seg)
    test = data_dir + 'test'
    test_tfidf = vector_space(test, stop_words=stpwd, vocabulary=train_tfidf.vocabulary, encoding=encoding, seg=seg)
    writeobj(os.path.join(save_path, 'train_tfidf.data'), train_tfidf)
    writeobj(os.path.join(save_path, 'test_tfidf.data'), test_tfidf)
    writeobj(os.path.join(save_path, 'vocabulary.data'), train_tfidf.vocabulary)


if __name__ == '__main__':
    data_dir = './SogouCS/'

    # 构建词袋
    tfidf_space(data_dir, data_dir + 'fearture_space', stopword_path=data_dir + 'stop_words.txt', seg=True)
```


## 二、分类算法

### 1. 重构代码

将一些工具函数转移到一个函数模块，func_tools.py :
```python
"""
工具函数
"""
import os
import pickle


def readfile(filepath, encoding='utf-8'):
    # 读取文本
    with open(filepath, "rt", encoding=encoding) as fp:
        content = fp.read()
    return content


def savefile(savepath, content):
    # 保存文本
    with open(savepath, "wt") as fp:
        fp.write(content)


def writeobj(path, obj):
    # 持久化python对象
    with open(path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def readobj(path):
    # 载入python对象
    with open(path, "rb") as file_obj:
        obj = pickle.load(file_obj)
    return obj


def check_dir_exist(dir):
    # 坚持目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)
```


### 2. 分类器

编写一个分类器类，可以根据需要选择不同的分类算法，初始化特征数据目录和模型保存文件名。若模型存在，则直接加载模型得到分类器，否则用训练集训练模型并保存。validation() 函数使用测试集验证模型，predict() 函数应用模型进行预测，可以传入文本所在目录给text_dir 参数，返回文件名与预测类别对应的列表。当预测一个目录下的文本时，enconding 参数可以设置文本的编码格式。也可以传入文本内容字符串给text_string参数，直接预测

分类器代码如下：
```python
"""
文本分类
实现读取文本，实现分词，构建词袋，保存分词后的词袋。
提取 tfidf 特征，保存提取的特征
"""
import os
from sklearn.externals import joblib
from sklearn import metrics
import func_tools as ft
from tfidf_feature import vector_space
from sklearn.feature_extraction.text import TfidfVectorizer


class TextClassifier:

    def __init__(self, clf_model, data_dir, model_path):
        """
        分类器
        :param clf_model:   分类器算法模型
        :param data_dir:    特征数据存放位置
        :param model_path:  模型保存路径
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.train_data = os.path.join(data_dir, 'train_tfidf.data')
        self.test_data = os.path.join(data_dir, 'test_tfidf.data')
        self.vocabulary_data = os.path.join(data_dir, 'vocabulary.data')
        self.clf = self._load_clf_model(clf_model)

    def _load_clf_model(self, clf_model):
        if os.path.exists(self.model_path):
            print('loading exists model...')
            return joblib.load(self.model_path)
        else:
            print('training model...')
            train_set = ft.readobj(self.train_data)
            clf = clf_model.fit(train_set.tdm, train_set.label)
            joblib.dump(clf, self.model_path)
            return clf

    def _predict(self, tdm):
        """
        :param tdm:     # 特征矩阵
        :return:
        """
        return self.clf.predict(tdm)

    def validation(self):
        """使用测试集进行模型验证"""
        print('starting validation...')
        test_set = ft.readobj(self.test_data)
        predicted = self._predict(test_set.tdm)
        actual = test_set.label
        for flabel, file_name, expct_cate in zip(actual, test_set.filenames, predicted):
            if flabel != expct_cate:
                print(file_name, ": 实际类别:", flabel, " --> 预测类别:", expct_cate)
        print('准确率: {0:.3f}'.format(metrics.precision_score(actual, predicted, average='weighted')))
        print('召回率: {0:0.3f}'.format(metrics.recall_score(actual, predicted, average='weighted')))
        print('f1-score: {0:.3f}'.format(metrics.f1_score(actual, predicted, average='weighted')))

    def predict(self, text_dir=None, text_string=None, encoding='utf-8'):
        """应用模型预测"""
        vocabulary = ft.readobj(self.vocabulary_data)
        if text_dir:
            tfidf_bunch = vector_space(corpus_dir=text_dir, stop_words=None, vocabulary=vocabulary, encoding=encoding, seg=True, tier=1)
            return list(zip(tfidf_bunch.filenames, self._predict(tfidf_bunch.tdm)))
        elif text_string:
            from tfidf_feature import tokenizer
            corpus = [' '.join(tokenizer().cut(text_string))]
            vectorizer = TfidfVectorizer(vocabulary=vocabulary)
            tdm = vectorizer.fit_transform(corpus)
            return self._predict(tdm)
        else:
            return None
```


### 3. 不同分类算法的比较

先用多项式贝叶斯算法训练模型：
```python
from sklearn.naive_bayes import MultinomialNB

data_dir = './SogouCS/'

clf = MultinomialNB(alpha=0.001)
model_path = data_dir + 'models/NBclassifier.pkl'

classifier = TextClassifier(clf, data_dir + '/fearture_space', model_path)
classifier.validation()
```
贝叶斯算法训练耗时很短，用测试集验证可以得到 0.921 的正确率：
```text
准确率: 0.921
召回率: 0.918
f1-score: 0.918
```

随机森林算法在测试集上有 0.924 的正确率：
```python
from sklearn.naive_bayes import MultinomialNB

data_dir = './SogouCS/'

clf = RandomForestClassifier(bootstrap=True, oob_score=True, criterion='gini')
model_path = data_dir + 'models/Radfclassifier.pkl'

classifier = TextClassifier(clf, data_dir + '/fearture_space', model_path)
classifier.validation()
```
用测试集验证可以得到 0.921 的正确率：
```text
准确率: 0.924
召回率: 0.925
f1-score: 0.924
```


Logistic 回归算法在测试集上能达到 0.972 的正确率：
```python
from sklearn.naive_bayes import MultinomialNB

data_dir = './SogouCS/'

clf = LogisticRegression(C=1000.0)
model_path = data_dir + 'models/LRclassifier.pkl'

classifier = TextClassifier(clf, data_dir + '/fearture_space', model_path)
classifier.validation()
```
用测试集验证可以得到 0.921 的正确率：
```text
准确率: 0.972
召回率: 0.972
f1-score: 0.972
```


### 4. 模型应用

```python
# # 模型应用
# 预测一个目录下的文本
ret = classifier.predict(text_dir=data_dir + 'tmp')
for item in ret:
    print(item)
# 预测一个文本字符串
text_string = '互联网一直在不经意中改变人们的生活和娱乐方式。当电视娱乐和纸介娱乐越来越同质化的时候，人们开始渴望一种新鲜的、更刺激的娱乐方式。于是，来自民间的智慧开始显现，网络恶搞应运而生，并迅速风靡中美这两个互联网最发达的国家。恶搞短片在带给人们无限快感的时候，也招来众多的批评。最新的消息称，国家广电总局将把视频纳入统一监管，引导视频带领中国互联网迈入一个新的时代。'
ret = classifier.predict(text_string=text_string)
print(ret)
```