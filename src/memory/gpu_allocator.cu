// #include "memory/gpu_allocator.h"
// namespace GcRT{
//     using namespace inner;
//     GPUAllocator::GPUAllocator(size_t bytes){
//         //支持动态扩容，如果容量不够了，申请Unified Memory作为替代
//         cudaMalloc(&dev_ptr_, bytes);
//     }

//     GPUAllocator::~GPUAllocator() noexcept{

//     }

//     std::optional<void *> GPUAllocator::Allocate(size_t bytes){
//         if(bytes < (1 << 16)){  //64KB,小于1个GPU分页大小，采用线程本地缓存方式 + Slab分配器
            
//         } else if(bytes < (1 << 24)){ //16MB以下，由伙伴系统处理
        
//         } else{  //太大的直接分配大页内存
        
//         }
//     }

//     void * GPUAllocator::AlignSize(size_t bytes){

//     }

//     void GPUAllocator::Defragment(){
        
//     }
// }