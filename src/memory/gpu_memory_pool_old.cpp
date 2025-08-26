// #include "memory/gpu_memory_pool.h"
// #include <cuda.h>
// #include <unordered_map>
// #include <thread>
// #include "rt_mutex.h"
// #include <list>
// #include <set>
// #include "data_structure/concurrentqueue.h"
// #include "logger/loggers.h"

// #ifdef DEBUG
// #include "debug/debug_logger.h"
// #endif

// namespace GcRT{
//     //allocate时，指针对应的CUDA流、大小需要记录，用BlockMeta来记录
//     //deallocate时，一般只传入一个void*，此时必须要查询BlockMeta才知道要回收多大的显存
//     struct BlockMeta {
//         size_t size;
//         bool is_active;
//         int class_idx;
//         cudaStream_t stream;  //对应的CUDA流编号
//         cudaEvent_t ready_event;  //空事件标记，异步操作中将本事件加入到对应的流，流中本事件执行完了才可以释放内存
//     };

//     struct AsyncFree {
//         void * ptr;
//         cudaEvent_t ready_event;
//     };

//     struct SizeClassPool
//     {
//         struct Chunk{  
//             void * dev_ptr;
//             mutable size_t used_count;    //每次allocate和deallocate，修改这里的used_count,defragment的时候，如果used_count == 0,
//         };

//         //chunk在集合内部应该按照指针顺序排序
//         struct ChunkSort{
//             bool operator()(const Chunk & c1, const Chunk & c2) const {
//                 return c1.dev_ptr < c2.dev_ptr;
//             }
//         };

//         std::set<Chunk, ChunkSort> chunks;
//         std::list<void * > free_list;    //TODO:这里本来考虑用红黑树替换，但是考虑到allocate是同步阻塞操作，要保证阻塞操作的性能

//         std::mutex allo_mutex;   //这里可以考虑在单个池子内部加锁，减少锁的粒度，本对象池内的所有对象基本都是并发不安全的

//         size_t chunk_size;  //整个chunk所占的字节数
//         size_t allocation_size;  //chunk每次可分配的字节数
//     };

//     class GPUMemoryPool::Impl{
//         using Chunk = SizeClassPool::Chunk;
//         using ChunkSort = SizeClassPool::ChunkSort;

//         static constexpr size_t kNumSmallClasses = 16;
//         static constexpr size_t kSmallSizes[kNumSmallClasses] = {    //最小分配page size为64
//             64, 128, 256, 512, 1024, 2048, 4096, 8192,
//             16384, 32768, 49152, 65536, 65536, 0, 0, 0  // 占位
//         };

//         static constexpr size_t kMaxSmallSize = 64 << 10;
//         static constexpr size_t kDefragment_interval = 100;   // 每执行kDefragment_interval次资源释放，进行一次碎片整理

//         static constexpr size_t refill_threshold = 50;  //本地缓存阈值
//         static constexpr size_t release_threshold = 200;    //本地释放阈值
//         //管理大小内存分配的块-全局
//         std::vector<SizeClassPool> _size_class_pools;  //不同大小的内存块的池

//         struct ThreadCache {  //TODO:ThreadLocal的分配、释放、回归全局池的逻辑有待完成
//             // struct CacheItem{
//             //     void * ptr;
//             //     int class_idx;
//             // };

//             std::unordered_map<void *, int> cache_info;
//             std::vector<std::vector<void *>> size_class_cache;

//             ThreadCache():size_class_cache(kNumSmallClasses){ }
//         };

//         //线程的本地缓存
//         thread_local static ThreadCache _thread_local_cache;  //不用上锁的本地缓存线程

//         //回收指针采用异步的方式，维护元数据表，维护一个(异步回收队列)，和一个后台线程
//         std::unordered_map<void *, BlockMeta> _metadata;  //allocate的每个指针对应的元数据,释放时检索
//         std::list<AsyncFree> _free_q;

//         std::mutex _free_mutex;  //_free_mutex用来锁这个异步回收队列 
//         std::mutex _meta_mutex;   //_meta_mutex用来锁元数据表
//         std::condition_variable _free_cv;
//         std::thread _free_thread;  //后台线程，用于回收指针
        
//         std::atomic<bool> _destorying{false};  //资源销毁指示量
//         std::atomic<int> _defragment_count{0} ; //执行碎片处理的指示量，当达到某个数字后，执行碎片处理

//     public:
//         Impl(): _free_thread(&GPUMemoryPool::Impl::free_worker, this), _size_class_pools(kNumSmallClasses) {
//             for(size_t i = 0; i < kNumSmallClasses; ++i){
//                 if(kSmallSizes[i] == 0) break;
//                 _size_class_pools[i].allocation_size = kSmallSizes[i];
//                 _size_class_pools[i].chunk_size = kSmallSizes[i] * 1024;
//             }
//         }

//         ~Impl(){
//             _destorying.store(true);
//             _free_cv.notify_all();
//             if(_free_thread.joinable()) _free_thread.join();
//             cleanup();  //后台线程停止后，再执行cleanup.
//         }

//     private:
//         //后台回收线程状态机
//         void free_worker(){
//             while(!_destorying){
//                 {
//                     UniqueLock locker(_free_mutex);  //只做阻塞等待的作用
//                     _free_cv.wait_for(locker, std::chrono::milliseconds(100), [this]() -> bool {return !_free_q.empty() || _destorying ;});
//                 }

//                 if(_destorying) break;

//                 //处理悬挂的释放指针，因为有些指针可能正在使用，当指针的ready_event执行完毕时，就可以宣告释放了
//                 process_pending_frees();

//                 _defragment_count ++;
//                 _defragment_count = _defragment_count % kDefragment_interval;
//                 if(_defragment_count == 0) defragment();  //碎片合并
//             }
//         }

//         void process_pending_frees(){            
//             auto it = _free_q.begin();
//             while (it != _free_q.end()) {
//                 cudaError_t status = cudaEventQuery(it->ready_event);
                
//                 if (status == cudaSuccess) {
//                     // ready事件已完成，可以安全释放
//                     void* ptr = it->ptr;
//                     size_t block_size;
//                     // 从元数据中移除
//                     {
//                         LockGuard meta_locker(_meta_mutex);
//                         auto meta_it = _metadata.find(ptr);
//                         if (meta_it != _metadata.end()) {
//                             cudaEventDestroy(meta_it->second.ready_event);
//                             block_size = meta_it->second.size;
//                             _metadata.erase(meta_it);
//                         }
//                     }
                    
//                     //如果是大显存，直接析构
//                     if(block_size > kMaxSmallSize) cudaFree(ptr);
//                     else{  //小显存的话，重新放进对应chunk的free_list
//                         int class_idx = get_size_class_index(block_size);
//                         auto & pool = _size_class_pools[class_idx];
                        
//                         {
//                             LockGuard locker(pool.allo_mutex);
//                             pool.free_list.push_back(ptr);
//                             auto upper = std::upper_bound(pool.chunks.begin(), pool.chunks.end(), Chunk{ptr, 0}, ChunkSort());
//                             if(upper != pool.chunks.begin()){
//                                 (--upper)->used_count--;  //这里设置了mutable允许修改const对象的数据
//                                 //const_cast<Chunk *>(&*(--upper))->used_count--;  //这里set内的元素不允许修改，因此选择用const_cast去掉const修饰
//                             } else {
//                                 std::cerr << "unknown error, happened in inner codes" << std::endl;
//                             }
//                         }
//                     }
                    
//                     it = _free_q.erase(it);
//                 } else if (status == cudaErrorNotReady) {  
//                     // 流中的ready事件没有完成，就先留着，下一轮再释放
//                     ++it;
//                 } else {
//                     std::cerr << "cudaEventQuery error: " 
//                             << cudaGetErrorString(status) << "\n";
//                     it = _free_q.erase(it);
//                 }
//             }
//         }

//         inline int get_size_class_index(size_t block_size){  //TODO:这里其实违反了依赖倒置原则，只是因为我知道它是这个规律所以才这么写
//             // return std::max(0, (int)std::log(block_size + 1) - 6);
//                 // 处理超过最大小内存的情况
//             if (block_size > kMaxSmallSize) 
//                 return kNumSmallClasses; // 返回无效索引（调用方需处理）

//             int l = 0, r = kNumSmallClasses - 1, mid = 0; //二分查找寻找对应的index
//             // 搜索第一个 >= block_size 的规格
//             while(l <= r){
//                 mid = l + (r - l) / 2;
//                 if(kSmallSizes[mid] < block_size) l = mid + 1;
//                 else r = mid - 1;
//             }

//             return l; // fallback
//         }

//         //碎片处理，对于一整片内存，实现回收
//         //TODO:这里的逻辑设计还是感觉有点问题，怎么去处理大块的内存呢？怎么合并块呢？
//         void defragment(){
//             //对每一个Pool执行：遍历所有的Chunk，如果有Chunk的use_count = 0,那么cuda回收这个Chunk
//             for(auto & pool : _size_class_pools){
//                 LockGuard locker(pool.allo_mutex);
//                 auto chunk_itr = pool.chunks.begin();
//                 while(chunk_itr != pool.chunks.end()){
//                     if(chunk_itr->used_count){  //还有used_count，说明还有显存在使用
//                         chunk_itr++;
//                         continue;
//                     }

//                     auto itr = pool.free_list.begin();
//                     while(itr != pool.free_list.end()){
//                         auto ptr = *itr;
//                         if(ptr >= chunk_itr->dev_ptr && ptr < static_cast<void *>(static_cast<char *>(chunk_itr->dev_ptr) + pool.chunk_size)){
//                             itr = pool.free_list.erase(itr);
//                         } else itr++;
//                     }

//                     cudaFree(chunk_itr->dev_ptr);
//                     chunk_itr = pool.chunks.erase(chunk_itr);
//                 }
//             }
//         }
    
//     public:
//         void cleanup(){
//             //阻塞直至完全删除
//             LockGuard locker(_free_mutex);
//             while( !_free_q.empty()){
//                 process_pending_frees();
                
//                 defragment();
//             }
//         }

//         void * allocate(size_t size, cudaStream_t stream){
//             void * ptr = nullptr;

//             if(size <= kMaxSmallSize){
//                 int class_idx = get_size_class_index(size);

//                 //优先从thread_local里面找
//                 auto & local_pool = _thread_local_cache.size_class_cache[class_idx];
//                 if( !local_pool.empty() ){
//                     ptr = local_pool.back();
//                     local_pool.pop_back();
//                     return ptr;
//                 }

//                 auto & pool = _size_class_pools[class_idx];
//                 //小显存分配
//                 {
//                     LockGuard locker(pool.allo_mutex);
//                     //如果池子里面的free_list还有，那么直接分配
//                     if(!pool.free_list.empty()){
//                         ptr = pool.free_list.back();
//                         pool.free_list.pop_back();

//                         //修改chunk中的索引，时间复杂度仅是O(log n/1024)，因为一个chunk块有1024个小块
                        
//                         auto upper = std::upper_bound(pool.chunks.begin(), pool.chunks.end(), Chunk{ptr, 0}, ChunkSort());
//                         if(upper != pool.chunks.begin()){
//                             (--upper)->used_count++;
//                             // const_cast<Chunk *>(&*(--upper))->used_count++;  //这里set内的元素不允许修改，因此选择用const_cast去掉const修饰
//                         } else {
//                             std::cerr << "unknown error, happened in inner codes" << std::endl;
//                         }

//                     } else {
//                         //else的情况，申请一个大内存，这个大内存包含若干小内存，push进free_list
//                         void * big_block;
//                         if(auto err = cudaMalloc(&big_block, pool.chunk_size); err != cudaSuccess){
//                             std::cerr << "cudaMalloc failed for size: " << size << std::endl;
//                         }

//                         SizeClassPool::Chunk new_chunk{big_block, 1};
//                         size_t counts_per_chunk = pool.chunk_size / pool.allocation_size;
//                         for(int i = 1;i < counts_per_chunk; ++i){
//                             void * chunk_ptr = static_cast<char*>(big_block) + i * pool.allocation_size;
//                             pool.free_list.push_back(chunk_ptr);
//                         }  

//                         pool.chunks.insert(std::move(new_chunk));
//                         ptr = big_block;
//                     } 
//                 }

//                 cudaEvent_t event;
//                 cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
//                 //记录元数据信息
//                 {
//                     LockGuard meta_locker(_meta_mutex);
//                     _metadata[ptr] = BlockMeta{pool.allocation_size, true, class_idx, stream, event};
//                 }

//             } else {  //大显存分配
//                 if (auto err = cudaMalloc(&ptr, size); err != cudaSuccess) {
//                     std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) 
//                             << " size: " << size << " bytes\n";
//                     return nullptr;
//                 }        
                
//                 cudaEvent_t event;
//                 cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

//                 {
//                     LockGuard locker(_meta_mutex);
//                     _metadata[ptr] = BlockMeta{size, true, -1, stream, event};  //class_idx = -1表示大对象
//                 }
//             }

//             return ptr;
//         }

//         void deallocate(void * ptr){
//             if(!ptr) return;

//             //如果本地Cache记录了，那么说明是小显存，先放入本地Cache，当然本地Cache是有上限的，不能随便放
//             if(_thread_local_cache.cache_info.size() < release_threshold){
//                 if (auto itr = _thread_local_cache.cache_info.find(ptr); itr != _thread_local_cache.cache_info.end()){
//                     int class_idx = itr->second;
//                     _thread_local_cache.size_class_cache[class_idx].push_back(ptr);
//                     return;
//                 }
//             }

//             LockGuard locker(_meta_mutex);
//             //释放前应该确保ptr已经不被使用了，否则会出现未定义的问题
//             if(auto itr = _metadata.find(ptr); itr != _metadata.end()){
//                 BlockMeta & meta = itr->second;
//                 if(!meta.is_active) return;  //如果已经标记is_active为false了，就不再重复释放
//                 meta.is_active = false;

//                 if(meta.stream != 0) cudaEventRecord(meta.ready_event, meta.stream);
//                 else cudaEventRecord(meta.ready_event);
//                 {
//                     LockGuard free_locker(_free_mutex);
//                     _free_q.push_back({ptr, meta.ready_event});
//                 }

//                 _free_cv.notify_one();
//             }  // else 也就是这个指针没找到，那就先不管
//         }
//     };

//     thread_local GPUMemoryPool::Impl::ThreadCache GPUMemoryPool::Impl::_thread_local_cache;

//     GPUMemoryPool::GPUMemoryPool() : _impl(std::make_unique<Impl>()){}

//     GPUMemoryPool::~GPUMemoryPool(){
//         cleanup();
//     }

//     GPUMemoryPool& GPUMemoryPool::instance() {
//         static GPUMemoryPool instance;  //单例模式
//         return instance;
//     }

//     void * GPUMemoryPool::allocate(size_t size, cudaStream_t stream){
//         return _impl->allocate(size, stream);
//     }
    
//     void GPUMemoryPool::deallocate(void * ptr){
//         _impl->deallocate(ptr);
//     }

//     void GPUMemoryPool::cleanup(){
//         _impl->cleanup();
//     }
// }