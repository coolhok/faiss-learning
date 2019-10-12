# IVFPQ 内存计算
## IVFPQ 存储内容
```
1.倒排入口中心 nlist * dim  float
2.PQ量化器中心值 sub * dim float
3.预计算残差列表 nlist * m * sub +  m * sub float
4.向量PQ编码值 m char
5.向量 ids long long
```
## cpu/gpu 存储类型
### CPU版本
1.向量存储均用 std::vector() 其扩容规则为 *2
### GPU版本
1.DeviceVector  其扩容规则为 2^n 
2.DeviceTerson  其扩容规则为 2^n 
## 静态加载内存计算
以如下数据为例
数据集合 
数据长度 1024*1024 
数据维度 128 维

IVFPQ 参数
nlist 1024
切割份数 m 8
聚类中心 sub 2^8 = 256
### cpu
1.倒排入口中心 1024 * 128 float32 = 512kb
2.PQ量化器中心值 sub * dim float32 = 128kb
3.预计算残差列表 nlist * m * sub +  m * sub float32 = 2050kb
4.向量PQ编码值 datalen * m char = 8192kb
5.向量 ids datalen * m  long long = 8192 kb

=18.62 Mb

### GPU
1.倒排入口中心 1024 * 128 float32 = 4kb + 倒排L2 1204 float32 = 516kb
2.PQ量化器中心值 sub * dim float32 = 128kb
3.预计算残差列表 nlist * m * sub +  m * sub float32 = 2050kb
4.向量PQ编码值 datalen * m char = 8192kb
5.向量 ids datalen * m  long long = 8192 kb

=18.63 Mb

## 动态生成索引
无论GPU/CPU版本 其存储单元均存在 扩容情况 则相同数据量下 其内存表现为静态加载的 1~2 倍之间

