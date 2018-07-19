# pytorch_image_sentence_matching

image sentence matching with generative adversarial networks

## 数据预处理

- load images
> 图像的加载，图像大小为`3*m*n`;
- load captions
> 图像的caption是保存在json文件中;
- 方法
> 1. 下载数据集，包括图像和文本注释
> 2. 安装cocoapi
> 3. ``data = dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])``加载数据集
> 4. 加载得到的数据集格式：
>  ``image,captions = data[i]``，其中``captions``是一个list，长度为5
> 5. 将```data```转换为一个```[sentence, matching image, unmatching image]```的矩阵，长度为```82783*5```



## 问题记录
1. **error [xx.zip]: start of central directory not found;**
> 在linux下，zip命令只能解压4G以下的zip文件；安装7zip软件，解压。
