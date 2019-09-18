# ivfpq文件格式加载

## 文件布局
```
 -------------
|             |
|    magic    |
|    number   |
|             |
 -------------
|             |　　　　　　　　　　　　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
|             |                   |　　　　｜　　　　｜　　　　｜　　　　｜　　　　 ｜　　　　｜　　
|    ivf      |------------------>|　 D   ｜ ntotal|　dummy｜dummy  ｜trained | type　｜
|    header   |                   |　　　　｜　　　　｜　　　　｜　　　　｜　　 　　｜　　　　｜　　
|             |　　　　　　　　　　　　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
 -------------
|             |
|    nlist    |
|             |
 -------------
|             |
|   nprobe    |
|             |
 -------------
|             |　　　　　　　　　　　 ---------
|             |                   |　IxF2 ｜　　
|  IVF        |------------------>------------------------------------------------
|  quantizer  |                   | 　D   ｜ ntotal |dummy |dummy ｜trained | type | 　
|             |　　　　　　　　　　　 ------------------------------------------------
|             |                   |xb.size| xb.data| direct | direct_map |   
|             |　                  ---------------------------------------
 -------------
|             |
|  residual   |
|             |
 -------------
|             |
| code_size   |
|             |
 -------------
|             |　　　　　　　　　　　 -------------------------
|             |                   |　 D   ｜ M     | nbits ｜　　　
|  Product    |------------------>|------------------------------------
|  Quantizer  |                   | centroids.size |　centroids.data() |
|             |　　　　　　　　　　　 -------------------------------------
 -------------
|             |　　　　　　　　　　　 ---------------------------
|             |                   | ilar | nlist | code_size |
|  Inverted   |------------------>----------------------------------------
|  List       |                   | full ｜ ids_list.size |ids_list.data |
|             |　　　　　　　　　　　 ----------------------------------------
|             |                   |codes| ids|
|             |　                  -----------
|             |                   |codes| ids|
|             |　                  -----------
 -------------                     .........
```
## 基础宏定义
```
// will fail if we write 256G of data at once...
#define READVECTOR(vec) {                       \
        long size;                            \
        READANDCHECK (&size, 1);                \
        FAISS_THROW_IF_NOT (size >= 0 && size < (1L << 40));  \
        (vec).resize (size);                    \
        READANDCHECK ((vec).data (), size);     \
    }
#define READANDCHECK(ptr, n) {                                  \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);              \
        FAISS_THROW_IF_NOT_FMT(ret == (n),                      \
            "read error in %s: %ld != %ld (%s)",                \
            f->name.c_str(), ret, size_t(n), strerror(errno));  \
    }
```
## 文件magic number读取
读取文件头4个字节
```
uint32_t h;
READ1 (h);
```
### ivfpq分支
```
else if(h == fourcc ("IvPQ") || h == fourcc ("IvQR") ||
              h == fourcc ("IwPQ") || h == fourcc ("IwQR")) {

        idx = read_ivfpq (f, h, io_flags);

}

static IndexIVFPQ *read_ivfpq (IOReader *f, uint32_t h, int io_flags)
{
    bool legacy = h == fourcc ("IvQR") || h == fourcc ("IvPQ");

    IndexIVFPQR *ivfpqr =
        h == fourcc ("IvQR") || h == fourcc ("IwQR") ?
        new IndexIVFPQR () : nullptr;
    IndexIVFPQ * ivpq = ivfpqr ? ivfpqr : new IndexIVFPQ ();

    std::vector<std::vector<Index::idx_t> > ids;
    read_ivf_header (ivpq, f, legacy ? &ids : nullptr);
    READ1 (ivpq->by_residual);
    READ1 (ivpq->code_size);
    read_ProductQuantizer (&ivpq->pq, f);

    if (legacy) {
        ArrayInvertedLists *ail = set_array_invlist (ivpq, ids);
        for (size_t i = 0; i < ail->nlist; i++)
            READVECTOR (ail->codes[i]);
    } else {
        read_InvertedLists (ivpq, f, io_flags);
    }

    // precomputed table not stored. It is cheaper to recompute it
    ivpq->use_precomputed_table = 0;
    if (ivpq->by_residual)
        ivpq->precompute_table ();
    if (ivfpqr) {
        read_ProductQuantizer (&ivfpqr->refine_pq, f);
        READVECTOR (ivfpqr->refine_codes);
        READ1 (ivfpqr->k_factor);
    }
    return ivpq;
}
```
## 读取ivf header
```
static void read_ivf_header (
    IndexIVF *ivf, IOReader *f,
    std::vector<std::vector<Index::idx_t> > *ids = nullptr)
{
    read_index_header (ivf, f);
    READ1 (ivf->nlist);
    READ1 (ivf->nprobe);
    ivf->quantizer = read_index (f);
    ivf->own_fields = true;
    if (ids) { // used in legacy "Iv" formats
        ids->resize (ivf->nlist);
        for (size_t i = 0; i < ivf->nlist; i++)
            READVECTOR ((*ids)[i]);
    }
    READ1 (ivf->maintain_direct_map);
    READVECTOR (ivf->direct_map);
}
```
### 读取index header
```
 -------------
|             |　　　　　　　　　　　　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
|             |                   |　　　　｜　　　　｜　　　　｜　　　　｜　　　　 ｜　　　　｜　　
|    ivf      |------------------>|　 D   ｜ ntotal|　dummy｜dummy  ｜trained | type　｜
|    header   |                   |　　　　｜　　　　｜　　　　｜　　　　｜　　 　　｜　　　　｜　　
|             |　　　　　　　　　　　　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
 -------------
```
```
static void read_index_header (Index *idx, IOReader *f) {
    READ1 (idx->d);   # 读取数据维度   int
    READ1 (idx->ntotal);  # 读取总数据数目 long
    Index::idx_t dummy; 
    READ1 (dummy);     # dummy读取 long
    READ1 (dummy);     # dummy读取 long
    READ1 (idx->is_trained);  # 文件是否已训练 一般默认为true
    READ1 (idx->metric_type); # IVFPQ 为L2格式
    idx->verbose = false;  # 默认verbose为false
}
```
### nlist nprobe 读取
```
 -------------
|             |
|    nlist    |
|             |
 -------------
|             |
|   nprobe    |
|             |
 -------------
```
```
READ1 (ivf->nlist);  # 读取 nlist
READ1 (ivf->nprobe); # 读取 nprobe
```
### ivf quantizer读取
```
 -------------
|             |　　　　　　　　　　　 ---------
|             |                   |　IxF2 ｜　　
|  IVF        |------------------>------------------------------------------------
|  quantizer  |                   | 　D   ｜ ntotal |dummy |dummy ｜trained | type | 　
|             |　　　　　　　　　　　 ------------------------------------------------
|             |                   |xb.size| xb.data| direct | direct_map |   
|             |　                  ---------------------------------------
 -------------
```
```
if (h == fourcc ("IxFI") || h == fourcc ("IxF2")) {
    IndexFlat *idxf;
    if (h == fourcc ("IxFI")) idxf = new IndexFlatIP ();
    else                      idxf = new IndexFlatL2 (); # IVFPQ走L2分支
    read_index_header (idxf, f);                         # 同read_index_header 读取header基本信息
    READVECTOR (idxf->xb);                               # 读取倒排文件类中点
    FAISS_THROW_IF_NOT (idxf->xb.size() == idxf->ntotal * idxf->d);  # 参数校验 xb.size == 倒排入口数×数据维度 此处ntotal是 IVF数 即nlist大小
    // leak!
    idx = idxf;
}
```
 ### IVFpq不走改分支可忽略
```
if (ids) { // used in legacy "Iv" formats
    ids->resize (ivf->nlist);
    for (size_t i = 0; i < ivf->nlist; i++)
        READVECTOR ((*ids)[i]);
}
```
 ### direct_map读取
```
READ1 (ivf->maintain_direct_map);
READVECTOR (ivf->direct_map);
```

 ## residual code_size读取
```
 -------------
|             |
|  residual   |
|             |
 -------------
|             |
| code_size   |
|             |
 -------------
```
READ1 (ivpq->by_residual); # 读取是否初始化残差矩阵
READ1 (ivpq->code_size);   # code_size

## ProductQuantizer 读取
```
 -------------
|             |　　　　　　　　　　　 -------------------------
|             |                   |　 D   ｜ M     | nbits ｜　　　
|  Product    |------------------>|------------------------------------
|  Quantizer  |                   | centroids.size |　centroids.data() |
|             |　　　　　　　　　　　 -------------------------------------
 -------------
```
static void read_ProductQuantizer (ProductQuantizer *pq, IOReader *f) {
    READ1 (pq->d);
    READ1 (pq->M);
    READ1 (pq->nbits);
    pq->set_derived_values ();
    READVECTOR (pq->centroids);
}
```
### 基本参数读取
```
READ1 (pq->d);   # 维度
READ1 (pq->M);   # 切割份数
READ1 (pq->nbits); # nbites 
```
### pq参数初始化计算
```
void ProductQuantizer::set_derived_values () {
    // quite a few derived values
    FAISS_THROW_IF_NOT (d % M == 0);
    dsub = d / M; 
    byte_per_idx = (nbits + 7) / 8;
    code_size = byte_per_idx * M;
    ksub = 1 << nbits;
    centroids.resize (d * ksub);
    verbose = false;
    train_type = Train_default;
}
```
### PQ k-means 类中心
```
READVECTOR (pq->centroids); # size = d * 2^nbits*4
```
## InvertedLists 倒排 codes ids读取
 -------------
|             |　　　　　　　　　　　 ---------------------------
|             |                   | ilar | nlist | code_size |
|  Inverted   |------------------>----------------------------------------
|  List       |                   | full ｜ ids_list.size |ids_list.data |
|             |　　　　　　　　　　　 ----------------------------------------
|             |                   |codes| ids|
|             |　                  -----------
|             |                   |codes| ids|
|             |　                  -----------
 -------------                     .........

```
 static void read_InvertedLists (
        IndexIVF *ivf, IOReader *f, int io_flags) {
    InvertedLists *ils = read_InvertedLists (f, io_flags);
    FAISS_THROW_IF_NOT (!ils || (ils->nlist == ivf->nlist &&
                                 ils->code_size == ivf->code_size));
    ivf->invlists = ils;
    ivf->own_invlists = true;
}

uint32_t h;
READ1 (h);
....................
....................
....................
} else if (h == fourcc ("ilar") && !(io_flags & IO_FLAG_MMAP)) {
    auto ails = new ArrayInvertedLists (0, 0);
    READ1 (ails->nlist);
    READ1 (ails->code_size);
    ails->ids.resize (ails->nlist);
    ails->codes.resize (ails->nlist);
    std::vector<size_t> sizes (ails->nlist);
    read_ArrayInvertedLists_sizes (f, sizes);
    for (size_t i = 0; i < ails->nlist; i++) {
        ails->ids[i].resize (sizes[i]);
        ails->codes[i].resize (sizes[i] * ails->code_size);
    }
    for (size_t i = 0; i < ails->nlist; i++) {
        size_t n = ails->ids[i].size();
        if (n > 0) {
            READANDCHECK (ails->codes[i].data(), n * ails->code_size);
            READANDCHECK (ails->ids[i].data(), n);
        }
    }
    return ails;
```

### nlist code_size 读取
```
READ1 (ails->nlist);  # nlist 
READ1 (ails->code_size); # code_size =  2^nbits
```
## 数组初始化
数据读取完成后定长resize,避免std::vector扩容带来不必要的内存开销
```
ails->ids.resize (ails->nlist);
ails->codes.resize (ails->nlist);
```
### 读取倒排列表 ids codes长度
```
read_ArrayInvertedLists_sizes (f, sizes);
for (size_t i = 0; i < ails->nlist; i++) { # 遍历读取 倒排列表codes ids长度
    ails->ids[i].resize (sizes[i]);
    ails->codes[i].resize (sizes[i] * ails->code_size);
}

static void read_ArrayInvertedLists_sizes (
         IOReader *f, std::vector<size_t> & sizes)
{
    uint32_t list_type;
    READ1(list_type);
    if (list_type == fourcc("full")) {
        size_t os = sizes.size();
        READVECTOR (sizes);
        FAISS_THROW_IF_NOT (os == sizes.size());
    } else if (list_type == fourcc("sprs")) {
        std::vector<size_t> idsizes;
        READVECTOR (idsizes);
        for (size_t j = 0; j < idsizes.size(); j += 2) {
            FAISS_THROW_IF_NOT (idsizes[j] < sizes.size());
            sizes[idsizes[j]] = idsizes[j + 1];
        }
    } else {
        FAISS_THROW_MSG ("invalid list_type");
    }
}
```

### 读取倒排codes ids实际内容
codes.size() ==  ids.size() 且对应偏移量相同
```
for (size_t i = 0; i < ails->nlist; i++) {
    size_t n = ails->ids[i].size();
    if (n > 0) {
        READANDCHECK (ails->codes[i].data(), n * ails->code_size);
        READANDCHECK (ails->ids[i].data(), n);
    }
}
```