# N-Body-Simulation

This project was created in the course of a bachelor thesis. 

It contains code for n-body simulations that was used to analyze the runtime behavior of two n-body algorithms: 
The naive approach and the Barnes-Hut alogrithm. 

Both algorithms are implemented using SYCL and can be executed in parallel on GPUs and CPUs. 

## Features
- Parallel SYCL implementation of the naive algorithm 
- Parallel SYCL implementation of the Barnes-Hut algorithm
- Support for CPUs as well as GPUs from NVIDIA and AMD
- ParaView output for visualization

## Installation

The project supports two different SYCL implementations: [Open SYCL](https://github.com/OpenSYCL/OpenSYCL) and [DPC++](https://github.com/intel/llvm).
The project supports Linux.

### Building the project with Open SYCL

The following commands build the project with Open SYCL for CUDA and OpenMP.
Replace `sm_XX` with the [compute capability](https://developer.nvidia.com/cuda-gpus) of your GPU, e.g., `sm_75`.

```
mkdir build
cd build
cmake -DUSE_DPCPP=OFF -DHIPSYCL_TARGETS="cuda:sm_XX;omp.accelerated" -DCMAKE_BUILD_TYPE=release ..
make
```

To build the project for AMD GPUs, replace `cuda:sm_XX` with `hip:gfx_XXX` and replace `gfxXXX` according to your AMD GPU.

If you do not wish to build the project with support for CPUs with OpenMP, delete `;omp.accelerated`.

For further details, please refer to the [Open SYCL documentation](https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/using-hipsycl.md).

### Building the project with DPC++

The following commands build the project with DPC++ for CUDA.

```
mkdir build
cd build
cmake -DUSE_DPCPP=ON -DCMAKE_BUILD_TYPE=release ..
make
```

The following commands build the project with DPC++ for AMD GPUS.
Make sure that the `DEVICE_LIB_PATH` environment variable points the the location of `<path to amdgcn>/amdgcn/bitcode/`

```
mkdir build
cd build
cmake -DUSE_DPCPP=ON -DUSE_DPCPP_AMD=ON -DDPCPP_ARCH="gfxXX" -DCMAKE_BUILD_TYPE=release ..
make
```

Replace `gfxXXX` according to your AMD GPU.
For more information, please refer to the [DPC++ documentation](https://intel.github.io/llvm-docs/GetStartedGuide.html)

