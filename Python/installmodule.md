# python代码执行安装第三名模块或升级已安装模块

使用 subprocess 模块执行命令行命令, 当导入未安装的模块时，自动安装。
```python
import subprocess

try:
    import modulename
except ModuleNotFoundError :
    subprocess.call('pip install modulename', shell=True)
    import modulename
```

结合 pip 模块，可以实现升级已安装模块：

```python
from subprocess import call
# 对于python3.6，需要先在当前环境中启用 pip 模块
import ensurepip
ensurepip.bootstrap()   

import pip

for dist in pip.get_installed_distributions():    
    call("pip install --upgrade " + dist.project_name, shell=True)
```