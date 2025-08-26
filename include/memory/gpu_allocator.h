// #ifndef GPU_ALLOCATOR_H
// #define GPU_ALLOCATOR_H
// /**
//  * GPUAllocator实现了一个类BFC的多线程显存池
//  * 它负责两部分的内容，一部分是模型的权重和激活值的保存与管理
//  * 另一部分是将数据通过PinnedMemory无额外内存拷贝环节拷贝到GPU中
//  */
// #include <NvInfer.h>
// #include <optional>
// namespace GcRT{
//     namespace inner{
//     class GPUAllocator {    
//         void * dev_ptr_;
//     public:
//         GPUAllocator() = delete;
//         GPUAllocator(size_t bytes);  //应该支持动态扩容
//         ~GPUAllocator() noexcept;

//         std::optional<void *> allocate(size_t bytes, cudaStream_t stream);
//         void deallocate(void * ptr, cudaStream_t stream);


//     private:
//         struct MemoryChunk{
//             size_t chunk_size;
//             size_t user_size;
            

//         };    


//         void * AlignSize(size_t bytes) noexcept;

//         void Defragment();
//     };
//     }
// }
// #endif