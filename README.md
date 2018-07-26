# pytorch_image_sentence_matching

image sentence matching with generative adversarial networks

## 数据预处理
### flickr8k
> 根据 Flickr8k.token.txt将图像id和图像注释文本拆分
- load sentence embedding
> flickr8k_hglmm_30_ica_sent_vecs_pca_6000.mat文件中保存
> 数据大小：6000*40454（使用h5py加载会自动转置）
- load images
> 根据图像id（flickr8k_image_ids.mat）加载对应图片
> 数据大小：40454*1（自动转置后会变成1*40454）
### coco数据集
- load images
> 图像的加载，图像大小为`m*n*3`;
- load captions
> 图像的caption是保存在json文件中;
>
## 生成器网络设计
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
2.flickr8k数据集错误数据
> flickr8k数据集中，Flickr8k.token.txt中存在5个特例（6731-6735行），图片名错误（2258277193_586949ec62.jpg.1）