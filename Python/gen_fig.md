# Python 生成 GIF 文件

使用 Python 合成 gif 动态图，程序如下：
 > 原图片需具有相同大小

```
import imageio
import os
import os.path

def create_gif(gif_name, path, duration = 0.3):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    path :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''

    frames = []
    pngFiles = os.listdir(path)
    image_list = [os.path.join(path, f) for f in pngFiles]
    for image_name in image_list:
        # 读取 png 图像文件
        frames.append(imageio.imread(image_name))
    # 保存为 gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)

    return

def main():
    gif_name = 'created_gif.gif'
    path = 'D:\\CSD'   #指定文件路径
    duration = 0.5
    create_gif(gif_name, path, duration)

if __name__ == "__main__":
    main()
```