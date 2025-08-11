cmake_minimum_required (VERSION 3.10)
project (GcRT LANGUAGES CXX CUDA)

if (POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

IF (CMAKE_BUILD_TYPE STREQUAL Debug)        # 用IF控制，可以实现Release版本不输出debug信息
    ADD_DEFINITIONS(-DDEBUG)                # 不知道为什么DEBUG必须写成-DDEBUG
ENDIF()

set(boost_asio_DIR )

# 设置Boost根目录
set(BOOST_ROOT "D:/boost_1_88_0")
# set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)

# 查找Boost组件（添加asio组件）
find_package(Boost 1.88.0 REQUIRED COMPONENTS system thread filesystem date_time)

if(Boost_FOUND)
include_directories(${Boost_INCLUDE_DIRS})
endif()

# set cuda、TensorRT
set(TRT_DIR "D:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4")
include_directories(${TRT_DIR}/include)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++17;-g;-G;-gencode;arch=compute_75;code=sm_75)
enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${TRT_DIR}/lib)

# include: this project、cuda、TensorRT
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)

# .exe's output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# .lib's output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# add subdir
add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(benchmark)

add_subdirectory(test)