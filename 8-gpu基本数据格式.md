# faiss gpu 基本数据格式

# DeviceVector
gpu数组

## class结构
```
enum MemorySpace {
  /// Managed using cudaMalloc/cudaFree
  Device = 1,
  /// Managed using cudaMallocManaged/cudaFree
  Unified = 2,
  /// Managed using cudaHostAlloc/cudaFreeHost
  HostPinned = 3,
};
```
```
template <typename T>
class DeviceVector {
  T* data_;          // 数据存放  T为数据类型
  size_t num_;       // 数据长度
  size_t capacity_;  //数据容量
  MemorySpace space_;   //基本都为 Device
}

DeviceVector::DeviceVector(MemorySpace space = MemorySpace::Device)
    : data_(nullptr),
    num_(0),
    capacity_(0),
    space_(space) {
}
```

## 基本方法
### 数组扩容
```
  void realloc_(size_t newCapacity, cudaStream_t stream) {
    FAISS_ASSERT(num_ <= newCapacity);

    T* newData = nullptr;
    allocMemorySpace(space_, &newData, newCapacity * sizeof(T));
    CUDA_VERIFY(cudaMemcpyAsync(newData, data_, num_ * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream));
    freeMemorySpace(space_, data_);

    data_ = newData;
    capacity_ = newCapacity;
  }
```
### 数组容量计算
```
向上取 2^n
template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
  return (isPowerOf2(v) ? (T) 2 * v : ((T) 1 << (log2(v) + 1)));
}
```

### 添加数据
```
bool append(const T* d,  //host 数据
              size_t n,  //数据长度
              cudaStream_t stream, // cuda stream
              bool reserveExact = false) {
    bool mem = false;

    if (n > 0) {
      //重新计算长度 其规则为 next Highest Power Of 2
      size_t reserveSize = num_ + n;
      if (!reserveExact) {
        reserveSize = getNewCapacity_(reserveSize);
      }
     
      //capacity_ 不足则扩容
      mem = reserve(reserveSize, stream);

      int dev = getDeviceForAddress(d);
      //cuda stream 数据拷贝
      if (dev == -1) {
        CUDA_VERIFY(cudaMemcpyAsync(data_ + num_, d, n * sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_VERIFY(cudaMemcpyAsync(data_ + num_, d, n * sizeof(T),
                                    cudaMemcpyDeviceToDevice, stream));
      }
      num_ += n;
    }

    return mem;
  }
```
### 清空释放资源
```
  void clear() {
    freeMemorySpace(space_, data_);
    data_ = nullptr;
    num_ = 0;
    capacity_ = 0;
  }
```
### drive copy to host
```
  template <typename OutT>
  std::vector<OutT> copyToHost(cudaStream_t stream) const {
    FAISS_ASSERT(num_ * sizeof(T) % sizeof(OutT) == 0);

    std::vector<OutT> out((num_ * sizeof(T)) / sizeof(OutT));
    CUDA_VERIFY(cudaMemcpyAsync(out.data(), data_, num_ * sizeof(T),
                                cudaMemcpyDeviceToHost, stream));

    return out;
  }
```
### 返回数据地址
```
T* data() { return data_; }
```

# DeviceTensor
设备张量类 改类为 class Tensor 的一层包装
```
template <typename T, //张量类型
          int Dim,    // 维度 (e.g..) 3*3 为2维数组 
          bool InnerContig = false, 
          typename IndexT = int,
          template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class DeviceTensor : public Tensor<T, Dim, InnerContig, IndexT, PtrTraits> {
 public:
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;

 private:
  enum AllocState {
    /// This tensor itself owns the memory, which must be freed via
    /// cudaFree
    Owner,

    /// This tensor itself is not an owner of the memory; there is
    /// nothing to free
    NotOwner,

    /// This tensor has the memory via a temporary memory reservation
    Reservation
  };
  AllocState state_;
  MemorySpace space_;
  DeviceMemoryReservation reservation_;
}
```
## 基本方法
### 引用DeriveVector数据
引用 DeviceVector 析构时不释放内存
```
template <typename T, int Dim, bool InnerContig,
          typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
  DataPtrType data,
  std::initializer_list<IndexT> sizes,
  MemorySpace space) :
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(data, sizes),
    state_(AllocState::NotOwner),
    space_(space) {
}
```
# Tensor
张量数据类
```
/**
   Templated multi-dimensional array that supports strided access of
   elements. Main access is through `operator[]`; e.g.,
   `tensor[x][y][z]`.

   - `T` is the contained type (e.g., `float`)
   - `Dim` is the tensor rank
   - If `InnerContig` is true, then the tensor is assumed to be innermost
   - contiguous, and only operations that make sense on contiguous
   - arrays are allowed (e.g., no transpose). Strides are still
   - calculated, but innermost stride is assumed to be 1.
   - `IndexT` is the integer type used for size/stride arrays, and for
   - all indexing math. Default is `int`, but for large tensors, `long`
   - can be used instead.
   - `PtrTraits` are traits applied to our data pointer (T*). By default,
   - this is just T*, but RestrictPtrTraits can be used to apply T*
   - __restrict__ for alias-free analysis.
*/
template <typename T,  //数据类型
          int Dim,     //维度 
          bool InnerContig = false, //数组是否连续
          typename IndexT = int, //size/stride 数据类型
          template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class Tensor {
 public:
  enum { NumDim = Dim };
  typedef T DataType;
  typedef IndexT IndexType;
  enum { IsInnerContig = InnerContig };
  typedef typename PtrTraits<T>::PtrType DataPtrType;
  typedef Tensor<T, Dim, InnerContig, IndexT, PtrTraits> TensorType;

  protected:
  /// Raw pointer to where the tensor data begins
  DataPtrType data_;  //真是数据存放

  /// Array of strides (in sizeof(T) terms) per each dimension
  IndexT stride_[Dim];  //张量跨度长度 3*3 张量 stride_={3,1} &data_[0][1] - &data_[0][0] =1
                        //                                 &data_[1][1] - &data_[1][1] =3

  /// Size per each dimension
  IndexT size_[Dim]; //张量数据维度 3*3张量 则size_ = {3，3}
}
```

## 基本方法

### 张量初始化
```
template <typename T, int Dim, bool InnerContig,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::
Tensor(DataPtrType data, const IndexT sizes[Dim])
    : data_(data) {
  static_assert(Dim > 0, "must have > 0 dimensions");

  //张量维度赋值  (e.g..) 256 *1024 张量 sizes[Dim] = {256,1204}
  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
  }
  //张量跨度复制 (e.g..) 256 *1024 张量 stride_[Dim] = [256,1]
  //&data_[0][1] - &data_[0][0] =1
  //&data_[1][1] - &data_[1][1] =256
  stride_[Dim - 1] = (IndexT) 1;
  for (int i = Dim - 2; i >= 0; --i) {
    stride_[i] = stride_[i + 1] * sizes[i + 1];
  }
}

//同上
template <typename T, int Dim, bool InnerContig,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::
Tensor(DataPtrType data, std::initializer_list<IndexT> sizes)
    : data_(data) {
  GPU_FAISS_ASSERT(sizes.size() == Dim);
  static_assert(Dim > 0, "must have > 0 dimensions");

  int i = 0;
  for (auto s : sizes) {
    size_[i++] = s;
  }

  stride_[Dim - 1] = (IndexT) 1;
  for (int j = Dim - 2; j >= 0; --j) {
    stride_[j] = stride_[j + 1] * size_[j + 1];
  }
}
```

###张量深度拷贝 支持跨显卡拷贝
```
template <typename T, int Dim, bool InnerContig,
          typename IndexT, template <typename U> class PtrTraits>
__host__ void
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::copyTo(
  Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t,
  cudaStream_t stream) {
  // The tensor must be fully contiguous
  GPU_FAISS_ASSERT(this->isContiguous());

  // Size must be the same (since dimensions are checked and
  // continuity is assumed, we need only check total number of
  // elements
  GPU_FAISS_ASSERT(this->numElements() == t.numElements());

  if (t.numElements() > 0) {
    GPU_FAISS_ASSERT(this->data_);
    GPU_FAISS_ASSERT(t.data());

    int ourDev = getDeviceForAddress(this->data_);
    int tDev = getDeviceForAddress(t.data());

    if (tDev == -1) {
      CUDA_VERIFY(cudaMemcpyAsync(t.data(),
                                  this->data_,
                                  this->getSizeInBytes(),
                                  ourDev == -1 ? cudaMemcpyHostToHost :
                                  cudaMemcpyDeviceToHost,
                                  stream));
    } else {
      CUDA_VERIFY(cudaMemcpyAsync(t.data(),
                                  this->data_,
                                  this->getSizeInBytes(),
                                  ourDev == -1 ? cudaMemcpyHostToDevice :
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    }
  }
}
```