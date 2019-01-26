# python 实现将 pandas 数据和 matplotlib 绘图嵌入 html 文件

> 实现用 python 将 pandas 的 DataFrame 数据以及 matplotlib 绘图的图像保存为 HTML 文件

#### 实现原理
1. python 的 [lxml](http://lxml.de/index.html) 库的 [etree](http://lxml.de/tutorial.html) 模块可以实现解析 HTML 代码并写入 html 文件。如下所示：
```
from lxml import etree
root = """
<title>lxml example</title>
<h1>Hello lxml!</h1>
"""
html = etree.HTML(root)
tree = etree.ElementTree(html)
tree.write('hellolxml.html')
```
2. pandas 的 DataFrame 数据，可直接调用 [df.to_html()](http://pandas.pydata.org/pandas-docs/stable/io.html#io-html) 函数将 DataFrame 数据转为 HTML 代码字符串。
3. 从 HTML 文件中可以发现，内嵌的图片是以 base64 代码的形式嵌入的。具体形式为 `<img src="data:image/png;base64,iVBORw... `。后面的 iVBORw...即为图像的 Base64 编码信息。故而只需将图像转为 base64 代码即可将图像嵌入 HTML 代码字符串中。python 的 base64 模块可以实现将二进制文件转为 base64 编码，而 matplotlib 的 [pyplot.savefig()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig) 函数可以将绘图窗口保存为二进制文件格式。

#### python 代码实现
最终便可使用 python 实现将将 pandas 的 DataFrame 数据以及 matplotlib 绘图的图像保存为 HTML 文件，代码如下。（以下代码基于 python 3.6 实现的。）
```
# 导入所需模块
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from lxml import etree
import base64
import urllib


# 获取数据集，用 urllib 库下载 iris 数据集作为示例
url = "http://aima.cs.berkeley.edu/data/iris.csv"
setl = urllib.request.Request(url)
iris_p = urllib.request.urlopen(setl)
iris = pd.read_csv(iris_p, sep=',',decimal='.',header=None, names=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species'])

# pandas 的 DataFrame 数据直接装换为 html 代码字符串
iris_des = """<h1>Iris Describe Stastic</h1>"""+iris.describe().T.to_html()

# matplotlib 任意绘制一张图
fig,axes = plt.subplots(1,4,sharey = True)
for n in range(4):
    axes[n].hist( iris.iloc[:,n],bins = 15,color = 'b',alpha = 0.5,rwidth= 0.8 )
    axes[n].set_xlabel(iris.columns[n])
plt.subplots_adjust(wspace = 0)
# figure 保存为二进制文件
buffer = BytesIO()
plt.savefig(buffer)  
plot_data = buffer.getvalue()

# 图像数据转化为 HTML 格式
imb = base64.b64encode(plot_data)  
#imb = plot_data.encode('base64')   # 对于 Python 2.7可用 
ims = imb.decode()
imd = "data:image/png;base64,"+ims
iris_im = """<h1>Iris Figure</h1>  """ + """<img src="%s">""" % imd   

root = "<title>Iris Dataset</title>"
root = root + iris_des + iris_im  #将多个 html 格式的字符串连接起来

# lxml 库的 etree 解析字符串为 html 代码，并写入文件
html = etree.HTML(root)
tree = etree.ElementTree(html)
tree.write('iris.html')

# 最后使用默认浏览器打开 html 文件
import webbrowser
webbrowser.open('iris.html',new = 1)
```