# index类型


## Index factory
```
用一个字符串构建Index，用逗号分割可以分为3部分：1.前处理部分；2.倒排表（聚类）；3.细化后处理部分

在前处理部分（preprocessing）：
1.PCA。"PCA64"表示通过PCA将数据维度降为64，"PCAR64"表示增加了随机旋转（random rotation）。
2.OPQ。"OPQ16"表示用OPQMatrix将数组量化为16位（待完善）
倒排表部分（inverted file）：
1."IVF4096"表示建立一个大小是4096的倒排表，即聚类为4096类。 细化部分（refinement）：
1."Flat"保存完整向量，通过IndexFlat或者IndexIVFFlat实现；
2."PQ16"将向量编码为16byte，通过IndexPQ或者IndexIVFPQ实现；
```

## index比较
| 索引名                                                | 类名                  | index_factory         | 主要参数                                           | 字节数/向量             | 精准检索 | 备注                                                           | 索引名                                                | 类名                  | index_factory         |
| -------------------------------------------------------- | ----------------------- | --------------------- | ------------------------------------------------------ | ---------------------------- | -------- | ---------------------------------------------------------------- | -------------------------------------------------------- | ----------------------- | --------------------- |
| 精准的L2搜索                                        | IndexFlatL2             | "Flat"                | d                                                      | 4*d                          | yes      | brute-force                                                      | 精准的L2搜索                                        | IndexFlatL2             | "Flat"                |
| 精准的内积搜索                                    | IndexFlatIP             | "Flat"                | d                                                      | 4*d                          | yes      | 归一化向量计算cos                                         | 精准的内积搜索                                    | IndexFlatIP             | "Flat"                |
| Hierarchical Navigable Small World graph exploration     | IndexHNSWFlat           | "HNSWx,Flat"          | d, M                                                   | 4d + 8 M                    | no       | -                                                                | Hierarchical Navigable Small World graph exploration     | IndexHNSWFlat           | "HNSWx,Flat"          |
| 倒排文件                                             | IndexIVFFlat            | "IVFx,Flat"           | quantizer, d, nlists, metric                           | 4*d                          | no       | 需要另一个量化器来建立倒排                          | 倒排文件                                             | IndexIVFFlat            | "IVFx,Flat"           |
| Locality-Sensitive Hashing (binary flat index)           | IndexLSH                | -                     | d, nbits                                               | nbits/8                      | yes      | optimized by using random rotation instead of random projections | Locality-Sensitive Hashing (binary flat index)           | IndexLSH                | -                     |
| Scalar quantizer (SQ) in flat mode                       | IndexScalarQuantizer    | "SQ8"                 | d                                                      | d                            | yes      | 每个维度项可以用4 bit表示，但是精度会受到一定影响 | Scalar quantizer (SQ) in flat mode                       | IndexScalarQuantizer    | "SQ8"                 |
| Product quantizer (PQ) in flat mode                      | IndexPQ                 | "PQx"                 | d, M, nbits                                            | M (if nbits=8)               | yes      | -                                                                | Product quantizer (PQ) in flat mode                      | IndexPQ                 | "PQx"                 |
| IVF and scalar quantizer                                 | IndexIVFScalarQuantizer | "IVFx,SQ4" "IVFx,SQ8" | quantizer, d, nlists, qtype                            | d or d/2                     | no       | 有两种编码方式：每个维度项4bit或8bit               | IVF and scalar quantizer                                 | IndexIVFScalarQuantizer | "IVFx,SQ4" "IVFx,SQ8" |
| IVFADC (coarse quantizer+PQ on residuals)                | IndexIVFPQ              | "IVFx,PQy"            | quantizer, d, nlists, M, nbits                         | M+4 or M+8                   | no       | 内存和数据id（int、long）相关，目前只支持 nbits <= 8 | IVFADC (coarse quantizer+PQ on residuals)                | IndexIVFPQ              | "IVFx,PQy"            |
| IVFADC+R (same as IVFADC with re-ranking based on codes) | IndexIVFPQR             | "IVFx,PQy+z"          | quantizer, d, nlists, M, nbits, M_refine, nbits_refine | M+M_refine+4 or M+M_refine+8 | no       | -                                                                | IVFADC+R (same as IVFADC with re-ranking based on codes) | IndexIVFPQR             | "IVFx,PQy+z"          |

## 是否需要精确的结果
如果需要，应该使用“Flat” 只有 IndexFlatL2 能确保返回精确结果。一般将其作为baseline与其他索引方式对比，以便在精度和时间开销之间做权衡。
不支持add_with_ids，如果需要，可以用“IDMap, Flat”。
支持GPU。
```
import faiss
#数据
import numpy as np 
d = 512          #维数
n_data = 2000   
np.random.seed(0) 
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype('float32')

#ids, 6位随机数
ids = []
start = 999
for i in range(data.shape[0]):
    ids.append(start)
    start += 2
ids = np.array(ids)

#不支持 add_with_ids
index = faiss.index_factory(d, "Flat")
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)

# 支持自己定义的id
index = faiss.index_factory(d, "IDMap, Flat")
index.add_with_ids(data, ids)
dis, ind = index.search(data[:5], 10)
print(ind)  
```


## 关心内存开销
需要注意的是faiss在索引时必须将index读入内存。

如果不在意内存占用空间，使用“HNSWx”
如果内存空间很大，数据库很小，HNSW是最好的选择，速度快，精度高，一般4<=x<=64。不支持add_with_ids，不支持移除向量，不需要训练，不支持GPU。
```

import faiss
#数据
import numpy as np 
d = 512          #维数
n_data = 2000   
np.random.seed(0) 
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype('float32')

index = faiss.index_factory(d, "HNSW16") # HNSW8 HNSW24 等
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)


# 如果稍微有点在意，使用“Flat“
# 聚类之后将每个向量映射到相应的bucket。该索引类型并不会保存压缩之后的数据，而是保存原始数据，所以内存开销与原始数据一致。通过nprobe参数控制速度/精度。
# 支持GPU,但是要注意，选用的聚类操作必须也支持。

Flatindex = faiss.index_factory(d, "IVF32, Flat") # IVF32 IVF 聚类方法 32 类中心
Flatindex.train(data)
Flatindex.add(data)
dis, ind = Flatindex.search(data[:5], 10)
print(ind)
```
### 降维降低内存
如果保存全部原始数据的开销太大。可以使用如下方法，包含三个部分，
1.降维
2.聚类
3.scalar 量化，每个向量编码为8bit 不支持GPU
```
index = faiss.index_factory(d, "PCAR16,IVF50,SQ8")  #每个向量降为16维
index.train(data)
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)
```
### 极致压缩
使用"OPQx_y,...,PQx"

y需要是x的倍数，一般保持y<=d，y<=4*x。 支持GPU。
```
index = faiss.index_factory(d, "OPQ32_512,IVF50,PQ32")  
index.train(data)
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)
```

在高效检索的index中，聚类是其中的基础操作，数据量的大小主要影响聚类过程。

#### 如果小于1M， 使用"...,IVFx,..."
N是数据集中向量个数，x一般取值[4sqrt(N),16sqrt(N)],需要30x ～ 256x个向量的数据集去训练。

#### 如果在1M-10M，使用"...,IMI2x10,..."
使用k-means将训练集聚类为2^10个类，但是执行过程是在数据集的两半部分独立执行，即聚类中心有2^(2*10)个。

#### 如果在10M-100M，使用"...,IMI2x12,..."