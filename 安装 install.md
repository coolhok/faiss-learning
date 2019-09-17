# 安装

## 使用Anaconda安装

使用Anaconda安装使用faiss是最方便快速的方式，facebook会及时推出faiss的新版本conda安装包，在conda安装时会自行安装所需的libgcc, mkl, numpy模块。
faiss的cpu版本目前仅支持Linux和MacOS操作系统，gpu版本提供可在Linux操作系统下用CUDA8.0/CUDA9.0/CUDA9.1编译的版本。
注意，上面语句中的cuda90并不会执行安装CUDA的操作，需要提前自行安装

```
# CPU 版本
建议：
1.python 版本 3.5
2.faiss 1.50以上
conda install faiss-cpu=1.51 -c pytorch

# 注意，语句中的cuda9/10并不会执行安装CUDA的操作，需要提前自行安装 cuda
# GPU 版本
conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8
conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10
```

## 编译安装
编译安装需要2个步骤：
1. 编译C++文件
2. 编译Python


```
#推荐使用 configure 生成编译变量
./configure && make (&& make install) for the C++ library, 

cd python; make && make install for the python interface.
```

configure配置项说明:
- `./configure --without-cuda` 只编译cpu版本.
- `./configure --with-cuda=/path/to/cuda-10.1` 指定gpu路径
- `./configure --with-cuda-arch="-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_72,code=sm_72"` 指定cuda版本
- `./configure --with-python=/path/to/python3.7` 指定python版本路径
- `LDFLAGS=-L/path_to_mkl/lib/ ./configure`设置 MKL BLAS路径或者通过LD_LIBRARY_PATH设置 

BLAS/Lapack
-----------

BLAS/Lapack 需要通过 configure 设置生效,Faiss需要 Fortran 77 编译实现的接口同时需要头文件

There are several BLAS implementations, depending on the OS and
machine. To have reasonable performance, the BLAS library should be
multithreaded. See the example makefile.inc's for hints and examples
on how to set the flags, or simply run the configure script:

   `./configure`

To check that the link flags are correct, and verify whether the
implementation uses 32 or 64 bit integers, you can

  `make misc/test_blas`

and run

  `./misc/test_blas`