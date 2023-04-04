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

To build the project for AMD GPUs, replace `cuda:sm_XX` with `hip:gfxXXX` and replace `gfxXXX` according to your AMD GPU.

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
cmake -DUSE_DPCPP=ON -DUSE_DPCPP_AMD=ON -DDPCPP_ARCH="gfxXXX" -DCMAKE_BUILD_TYPE=release ..
make
```

Replace `gfxXXX` according to your AMD GPU.
For more information, please refer to the [DPC++ documentation](https://intel.github.io/llvm-docs/GetStartedGuide.html).

## Running the program

The programm has several optional and mandatory program arguments.

### Mandatory program arguments

| Argument          | Description         | Notes             |
| ----------------- | ------------------- | ----------------- |
| `--file` | Path to a .csv file containing the data for the simulation | - |
| `--dt` | Width of the time step for the simulation  | E.g.: `1h` for one hour |
| `--t_end` | The internal time until the system will be simulated | E.g.: `365d` for  365 days or `12y` for twelve years |
| `--vs` | The time step width of the visualization  | E.g.: `1d` to visualize every day |
| `--vs_dir` | The top-level output directory for the output files | A separete foulder (with a time stamp) that contains all <br /> output files will be created in this directory|
| `--algorithm` | The algorithm to use for the simulation  | Either `<naive>` or `<BarnesHut>` |

### Optional programm arguments

| Argument          | Description         | Notes             |
| ----------------- | ------------------- | ----------------- |
| `--use_gpus` | Enable / disable execution on GPUs if build for GPUs and CPUs <br />(GPU execution is enabled by default) | `true` or `false` |
| `--energy` | Enable / disable computation of the energy of the system in each <br /> visulaized step (disabled by default). <br /> The results will be written to the ParaView files. | `true` or `false`, <br /> can result into long runtimes with large datasets|


### Optional arguments for the naive algorithm

| Argument          | Description         | Notes             |
| ----------------- | ------------------- | ----------------- |
| `--block_size` | The size of the blocks after which the <br /> local memory is updated in the naive algorithm | Should be a power of 2, e.g., `128` |
| `--opt_stage` | Selects the optimization stage of the naive algorithm (2 is default) | Possible values:<br /> <ul><li>`0` (non-optimized)</li><li>`1`</li><li>`2` (highest degree of optimization)</li></ul> |

### Optional arguments for the Barnes-Hut algorithm
| Argument          | Description         | Notes             |
| ----------------- | ------------------- | ----------------- |
| `--theta` | The theta-value which determines the <br /> accuracy of the Barnes-Hut algorithm | Smaller values like `0.2` mean worse <br />  performance but better accuracy <br /> Larger values like `1` result into better <br />  performance but worse accuracy |
| `--num_wi_octree` | Determines the number of work-items <br /> used to build the octree | - |
| `--num_wi_top_octree` | Determines the number of work-items <br /> used to build the top of the octree | - |
| `--num_wi_AABB` | Determines the number of work-items <br /> used to calculate the AABB | - |
| `--num_wi_com` | Determines the number of work-items used <br /> to calculate the center of mass | - |
| `--max_level_top_octree` | Determines the maximum level <br /> to which the top of the octree gets build | - |
| `--wg_size_barnes_hut` | Determines the work-group size of the acceleration kernel | - |
| `--sort_bodies` | Enable / disable sorting of the bodies according to their <br /> in-order position in the octree (enabled by default) | `true` or `false`|
| `--storage_size_param` | Scales the amount of memory for the octree data structures | Use only if you encounter problems <br />  with specific datasets |
| `--stack_size_param` | Scales the amount of memory for the octree data structures | Use only if you encounter problems <br /> with specific datasets  |


