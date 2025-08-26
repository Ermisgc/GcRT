// /**
//  * @file arena.h
//  * @brief jemalloc中的arena概念，应用线程通过轮询的方式选择一个Arena进行内存分配
//  */
// #include <memory>
// #include <atomic>
// #include <mutex>
// #include <cmath>
// #include <vector>

// namespace GcRT{
//     using dptr_t = void *;
//     using size_t = unsigned int;
// namespace Memory{
//     using malloc_mutex_t = std::mutex;
//     class Arena{
//         static constexpr size_t kSuperblockSize = 4 << 20;
//         unsigned int index_;
//         std::atomic<int> app_threads_;
//         std::atomic<int> inner_threas_;

//     };


//     class ArenaChunk{


//     };

//     /**
//      * @brief 按显存请求将大小分为三种对象：小对象（<64KB）、中对象（64KB ~ 4MB）、大对象（>4MB）
//      * @ref 这里参考了Jemalloc的大小对象设置，Jemalloc是内存中的，PageSize = 4KB，
//      * 低于4KB的，分为小对象，4KB ~ 4MB的是中对象，而超过4MB的是大对象。
//      * 那这里作为GPU中的内存池，PageSize = 64KB
//      * 同理，低于64KB的我设置为小对象，64KB ~ 4MB的认为是中对象，超过4MB的是大对象
//      * 考虑一个大小为3 * 256 * 256的图片，其大小应该是3 * 64KB = 192KB，一个batch_size = 16的tensor正好可以占用一个中对象
//      * 因此考虑一般将batch_size设置为16
//      */
//     class SizeClass{
//     private:
//         /**
//          * GPU中单个页表的大小
//          */
//         static constexpr size_t kGPUPageSize = 64 << 10;

//         /**
//          * 三阶梯对象的分类标准
//          * kMinimumPageSize对应最小的分配单位，在GPU中为64B
//          * kMaxSmallSize是小对象和中对象的判断依据
//          * kMaxMediumSize是中对象和大对象的判断依据
//          */
//         static constexpr size_t kMinimumPageSize = 64;
//         static constexpr size_t kMaxSmallSize = 64 << 10;
//         static constexpr size_t kMaxMediumSize = 4 << 20;

//         /**
//          * 小对象的分组效仿slab分配器，按每2倍的分类方法，对应每个bin之间的两倍距离
//          */
//         static constexpr size_t kNumSmallClasses = 16;
//         static constexpr size_t kSmallSizes[kNumSmallClasses] = {  
//             64, 128, 256, 512, 1024, 2048, 4096, 8192,
//             16384, 32768, 49152, 65536, 65536, 0, 0, 0  // 占位
//         };
//     public:
//         /**
//          * @brief 获得某个尺寸在分类标准中属于哪一类对象:
//          * 大于kMaximum的是大尺寸对象
//          * 如果是中尺寸对象，选择按中对象的整体尺寸进行分类，看它是多少个页表，向上取整
//          * 小对象尺寸看它对应的bin的大小
//          */
//         static size_t Classify(size_t size){
//             if(size >= kMaxMediumSize){
//                 return size;  //大对象直接返回原始大小
//             } else if(size >= kMaxSmallSize){
//                 return (size + kGPUPageSize - 1) / kGPUPageSize; //中对象返回页数，向上取整
//             } else {
//                 return kSmallSizes[std::max(0, (int)std::log(size + 1) - 6)];
//             }
//         }
//     };

//     /**
//      * @brief 超级块，借鉴jemalloc的extent概念，它是一个大小（total_size）超过kMaxMediumSize的空间的管理单元，
//      * 因此可以在一个超级块内实现对中、小对象的管理，具体的管理对象大小用（block_size）来表示。
//      * 它由arena管理，在中内存分配中充当buddy算法的chunk,在小内存分配中充当slab。
//      * 它的作用主要是尽可能地按块分配，按块释放，从而可以减少内存碎片。
//      */
//     class SuperBlock{
//         dptr_t gpu_ptr_;
//         size_t total_size_;
//         size_t block_size_;
        
//         /**
//          * 剩余可分配对象的数量，一开始等于total_size_ / block_size_
//          */
//         size_t free_count_;

//         /**
//          * 选择用位图而不是空闲链表来表示每个block被分配与否
//          */
//         std::vector<uint8_t> bitmap_;  

//     public:
//         SuperBlock() = delete;
//         /**
//          * 根据超级块的大小和管理的对象的大小来创建superblock，
//          * 这里需要传入一个来自于Arena创建的指针
//          */
//         SuperBlock(void * ptr, size_t size, size_t blk_size):gpu_ptr_(ptr), total_size_(size), block_size_(blk_size),
//         free_count_(size / blk_size), bitmap_(free_count_, false){

//         }

//         /**
//          * 遍历位图，查找到第一个没有被分配的内存块，时间为O(n)，n是块数
//          */
//         dptr_t AllocateBlock(){
//             for(size_t i = 0;i < bitmap_.size(); ++i){
//                 if(!bitmap_[i]){
//                     bitmap_[i] = true;
//                     free_count_--;
//                     return static_cast<char *>(gpu_ptr_) + i * block_size_;  //void *型指针不能进行基地址加减，因此考虑转化为char*
//                 }
//             }            
//             return nullptr;
//         }

//         /**
//          * 释放块，然后将位图还原
//          */
//         bool FreeBlock(dptr_t ptr){
//             size_t offset = (static_cast<char *>(ptr) - static_cast<char *>(gpu_ptr_));
//             if(offset % block_size_ || offset >= total_size_ || offset < 0){
//                 return false;
//             }

//             size_t index = offset / block_size_;
            
//             //块已经被回收了
//             if(!bitmap_[index]) return false;
//             bitmap_[index] = 0;
//             free_count_ ++;
//             return true;
//         }
//     };



// }

// }