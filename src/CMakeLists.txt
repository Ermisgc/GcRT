# aux_source_directory(. SRC_LIST)
file(GLOB_RECURSE SRC_LIST "*.cpp" "*.c" "*.cu" "*.cc")
add_library(gcrt ${SRC_LIST})
target_link_libraries(gcrt nvinfer_10 nvonnxparser_10 CUDA::cudart)