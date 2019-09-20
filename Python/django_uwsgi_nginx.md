# Django + uWSGI + nginx 部署 Python Web 应用

在实现一些算法模式后，考虑的就是模型的部署了。对于一些相对独立的模型应用，如 OCR 识别引擎等，部署为独立的 Web 应用，以提供 API 的供其他系统调用的方式将是一个不错的解决方案。此方案的优点有以下几个方面：

- 模型应用与其他系统隔离，仅通过 web 请求调用，既能极大得扩展应用的兼容性，又便于模型的迭代升级。
- 当 web 应用启动后，模型文件加载到内存，每次运行模型都无需重新载入，大大缩短调用时间。

## 本文使用的环境

- ubuntu-16.04.3-server
- Anaconda3 的 Python 3.7
- Django-2.2.5
- uwsgi-2.0.18
- nginx-1.14.0

## 1、Django 的安装和使用

- 安装
```
pip install Django
```

- 创建项目
```
django-admin startproject demosite
```
将会在当前目录下创建一个web项目，项目目录结构如下：
```
demosite/
├── demosite
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── manage.py
```

在 demosite/demosite/ 目录下创建 views.py 文件，输入代码：
```
from diango.http import HttpResponse


def hello(request):
    return HttpResponse("Hello World! [CSD]")
```

进入 urls.py 文件中添加：
```
from demosite.views import hello
```
在 urlpatterns 中添加 url 路径：
```
urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', hello),
]
```
修改 settings.py 文件，将本地的IP添加到 ALLOWED_HOSTS = [], 如：
```
ALLOWED_HOSTS = ['192.168.0.100'] 
```

- 启动项目

在 demosite 目录下，运行：
```
python manage.py runserver 192.168.0.100:8000
```

然后在浏览器中输入 url：
```
http://192.168.0.100:8000/hello/
```

就可以看的返回的
```
Hello World! [CSD]
```

Django 更高级的使用和配置可以参考官方文档：[https://docs.djangoproject.com/en/2.2/](https://docs.djangoproject.com/en/2.2/)


## 2、使用 uWSGI 管理 Django 应用

使用 uWSGI 就可以部署一个简单的 Django Web 应用，管理 WEB 请求实现良好的并发访问。同时 uWSGI 是 Django 推荐的方式: [How to use Django with uWSGI](https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/uwsgi/#how-to-use-django-with-uwsgi)，详情可参考 [uWSGI 的官网](https://uwsgi-docs.readthedocs.io/en/latest/index.html).

- 安装 uWSGI

安装 uWSGI 需要 c 编译器，安装方法参考官方的[安装指南](https://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html)，也可以用 conda 直接安装：

```
conda install -c conda-forge uwsgi
```

- 使用 uwsgi 启动 Django 应用

在 demosite 目录下(也可以在其他目录)新建 demo_uwsgi.ini 配置文件：
```
[uwsgi]
chdir = /home/csd/demosite
wsgi-file = demosite/wsgi.py
buffer-size = 65536
master = true
processes = 8
threads=10
http = 192.168.0.100:8080  
vaccum = true
daemonize = /home/csd/demosite/logs/demo_uwsgi.log
```
在 demosite 目录下创建文件夹：logs/

启动命令如下：
```
uwsgi --ini demo_uwsgi.ini --enable-threads
```
`--enable-threads` 参数用于启动多线程支持。

此时可以访问链接：
```
http://192.168.0.100:8080/hello/
```

此时关闭项目需要执行：
```
killall -9 uwsgi
```

- 安装 libiconv

如果上面启动 uwsgi 报错：
```
uwsgi: error while loading shared libraries: libiconv.so.2: cannot open shared object file: No such file or directory
```

则需要安装 libiconv 
```
wget http://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.14.tar.gz
tar zxvf libiconv-1.14.tar.gz
cd libiconv-1.14
./configure
sudo make
sudo make install
```
如果 make 出错：
```
/stdio.h:1010:1: error: ‘gets’ undeclared here (not in a function) 
_GL_WARN_ON_USE (gets, “gets is a security hole - use fgets instead”); 
```
则在 libiconv-1.14/srclib/stdio.in.h 文件中，将698行的代码：
```
_GL_WARN_ON_USE (gets, "gets is a security hole - use fgets instead");
```
替换为：
```
#if defined(__GLIBC__) && !defined(__UCLIBC__) && !__GLIBC_PREREQ(2, 16)
_GL_WARN_ON_USE (gets, "gets is a security hole - use fgets instead");
#endif
```
然后从新执行：
```
./configure
sudo make
sudo make install
```

安装完成后，切换 root 账户，输入以下命令：
```
echo '/usr/local/lib' >> /etc/ld.so.conf
ldconfig
```
最后输入：
```
iconv --version
```
如果能执行则安装正确。



## 3、nginx 部署

使用 nginx 服务器部署，实现负载均衡，更好得管理 WEB 请求。

- 安装

需要下载以下几个安装包：
```
nginx-1.14.0.tar.gz  
nginx_uploadprogress_module-0.9.0.tar.gz  
openssl-1.0.2o.tar.gz  
pcre-8.42.tar.gz
```
解压后进入 nginx-1.14.0/ 编译并安装 nginx：
```
./configure --prefix=/home/csd/nginx --with-openssl=../openssl-1.0.2o --with-pcre=../pcre-8.42 --add-module=../masterzen-nginx-upload-progress-module-a788dea --without-http_gzip_module
make
make install 
```

- 配置

进入安装目录 (/home/csd/nginx/) 下的 conf 目录，修改 nginx.conf 配置文件

将工作进程设置为4：
```
worker_processes  4;
```
nginx 默认监听 80 端口，所以有必要先查看 80 端口是否已被占用：
```
netstat -lnp|grep 80
```
如果被占用则改为其他端口，server 配置中增加如下配置：
```
server {
    listen       90;
    server_name  192.168.0.100;
    large_client_header_buffers 4 16k;
    client_max_body_size 300m;
    client_body_buffer_size 128k;
    proxy_connect_timeout 600;
    proxy_read_timeout 600;
    proxy_send_timeout 600;
    proxy_buffer_size 256k;
    proxy_buffers   16 256k;
    proxy_busy_buffers_size 512k;
    proxy_temp_file_write_size 512k;

    location / {
        root   html;
        index  index.html index.htm;
        include     uwsgi_params;
        uwsgi_pass   unix:/home/csd/demosite/uwsgi.sock;
    }
    location /static {
        alias /home/csd/demosite/static;
    }
    location /media {
        alias /home/csd/demosite/media;
    }
}
```

接着在 demosite 目录下创建 `static/` 和 `media/` 两个文件夹。将 `demo_uwsgi.ini` 中的 
```
http = 192.168.0.100:8080
``` 
修改为 
```
socket =  /home/csd/demosite/uwsgi.sock
```
在 `settings.py` 文件最后添加：
```
STATIC_ROOT = os.path.join(BASE_DIR, "static/")
```

- 启动服务

收集静态文件，在 demosite 目录下运行：
```
python manage.py collectstatic
```
启动 uwsgi：
```
uwsgi --ini demo_uwsgi.ini --enable-threads
```
启动 nginx：
```
sudo /home/csd/nginx/sbin/nginx
```
此时访问：
```
http://192.168.0.100:90/hello/
```
