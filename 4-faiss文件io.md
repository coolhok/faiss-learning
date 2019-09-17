# faiss index 写入文件

## index文件读写
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

#index 写入文件
faiss.write_index(index, "./test.faiss") #将index保存为index_file.index文件

#index 从文件读取index
ioindex = faiss.read_index("./test.faiss")
print(ioindex.d)
```
## faiss还提供以下文件接口
```
# 写入
void write_index_binary (const IndexBinary *idx, const char *fname);
void write_index_binary (const IndexBinary *idx, FILE *f);
void write_index_binary (const IndexBinary *idx, IOWriter *writer);

# 读取
IndexBinary *read_index_binary (const char *fname, int io_flags = 0);
IndexBinary *read_index_binary (FILE * f, int io_flags = 0);
IndexBinary *read_index_binary (IOReader *reader, int io_flags = 0);
```

## index clone
```
# 完全复制一个index
index_new = faiss.clone_index(index)
```