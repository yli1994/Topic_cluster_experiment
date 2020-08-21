# 文本主题聚类

## **目标：尝试不同的句向量和聚类方法带来的效果** 以及TSNE降维可视化  

### 一、词向量下载地址:   

链接: https://pan.baidu.com/s/1GerioMpwj1zmju9NkkrsFg 
提取码: x6v3
请下载之后在项目根目录建立"embeddings"文件夹, 将下载的文件放入(不用解压), 即可运行代码.   

### 二、目前尝试方法：

词向量-->句向量目前尝试了两种方法:均值法和SIF方法(https://github.com/PrincetonML/SIF)

聚类：Kmeans和GMM， 后续会加入别的方法

(数据集是一个轻量的情感分类数据集，标签对聚类实际没有用处，随便找的)

### 三、方法SIF+Kmeans结果：

![SIF_Kmeans](results\SIF_Kmeans.png)