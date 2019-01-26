# python 的 IDLE 无法连接以及 jupyter notebook 无法打开浏览器

### 问题描述
win10 系统由于防火墙机制，安装 python 以及 anaconda 会出现如下错误：
1. 安装 python 后，打开 IDLE 出现错误，无法连接python解释器。错误信息为
IDLE's subprocess didn't make connection.Either IDLE can;t
start a subprocess or personal firewall software is blocking
the connection. 

2. 安装 Anaconda 后，发现 jupyter notebook 无法打开浏览器。notebook 无法打开的问题也会出现在原本可以打开但一段时间后又打不开的情况，在 win7 系统也会出现这种问题。

以上问题是防火墙拦截导致的。

### 解决办法
解决办法是将python添加到防火墙例外：

**控制面板 -> 系统和安全 -> Windows Defender 防火墙 -> 允许的应用**

点击**允许其他程序**，找到安装目录下，将python.exe 和pythonw.exe添加到允许列表，专用和公用均打钩。