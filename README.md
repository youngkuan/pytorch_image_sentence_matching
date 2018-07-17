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
> 3. ``dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])``加载数据集