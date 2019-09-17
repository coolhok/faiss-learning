#faiss learn 学习文档
本仓部分资料是直接搬运过来
部分材料是自己整理

https://github.com/facebookresearch/faiss/wiki

## faiss版本
faiss-1.5.0

## 其他资料链接

### LSH(Local Sensitive Hashing)
https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Primer

### PQ论文
https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf

### IVFPQ python 实现：
https://github.com/matsui528/rii

### 高纬度向量搜索性能横向比对
https://github.com/erikbern/ann-benchmarks

### 百万级别向量检索 Indexing 1M vectors
当数据集中的向量个数在百万级别，暴力精确搜索的时间开销太大，比较好的选择是使用IndexIVFFlat索引类型。IndexIVFFlat也会返回精确的距离值，但返回的结果并不是完全正确的，可能会漏掉某个结果。
facebook官方通过一些实验，通过不同的检索类型在1百万向量的数据集上做检索，其中主要关注速度和精度的变化。实验结果展示在faiss wiki中。
实验中使用特征提取器提取1百万张图片的特征表达，对每张图片提取4096维特征向量，然后使用PCA将4096维向量降维到256维。

faiss wiki:https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors

### 1G级别向量检索 Indexing 1G vectors
对这一量级的数据集，必须使用向量的压缩编码形式，主要的方法有乘积量化（PQ）。
使用Bigann和Deep1B分别进行实验。实验结果在faiss wiki。
faiss wiki:https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors
