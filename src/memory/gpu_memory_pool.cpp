#include "memory/gpu_memory_pool.h"
#include <cuda.h>
#include <unordered_map>
#include <thread>
#include "rt_mutex.h"
#include <list>
#include <set>
#include <queue>
#include "data_structure/concurrentqueue.h"
#include "logger/loggers.h"

#ifdef DEBUG
#include "debug/debug_logger.h"
#endif

/**
 * @file 
 * //TODO:大内存分配和中内存分配还未实现，考虑后续采用Buddy System重新优化中内存分配 
 */

namespace GcRT{

    //allocate时，指针对应的CUDA流、大小需要记录，用BlockMeta来记录
    //deallocate时，一般只传入一个void*，此时必须要查询BlockMeta才知道要回收多大的显存
    struct BlockMeta {
        size_t size;
        bool is_active;
        int class_idx;
        cudaStream_t stream;  //对应的CUDA流编号
        cudaEvent_t ready_event;  //空事件标记，异步操作中将本事件加入到对应的流，流中本事件执行完了才可以释放内存
    };

    //异步释放信息，通过判断ready_event是否完成了，来判断指针是否还在用
    struct AsyncFree {
        void * ptr;
        int class_idx;
        cudaEvent_t ready_event;
        bool is_huge = {false};
    };
    // using AsyncFree = BlockMeta;

    struct Chunk{  //Chunk本身不支持释放，释放工作交给Center Cache 
        void * dev_ptr;
        size_t chunk_size;  //整个chunk所占的字节数
        int class_idx;  //chunk让谁分去了
        size_t allocated_size;  //chunk总共分为了多少份
        //每次allocate和deallocate，修改这里的used_count,defragment的时候，如果used_count == 0,那么这个块实际上可以回收
        //但是要注意CAS操作，先锁定0，然后改为-1，如果这个过程失败了，说明有别的线程抢占了used_count，修改了它的值
        mutable std::atomic<int> used_count;    
        mutable std::atomic<bool> reclaiming;

        Chunk(void *p , size_t s, int class_i, size_t as, int u, bool r = false) : dev_ptr(p), chunk_size(s), class_idx(class_i),\
            allocated_size(as), used_count(u), reclaiming(r){}

        Chunk(void * p): dev_ptr(p), used_count(0), reclaiming(false) {}

        Chunk(const Chunk & c){
            //拷贝构造
            dev_ptr = c.dev_ptr;
            chunk_size = c.chunk_size;
            class_idx = c.class_idx;
            allocated_size = c.allocated_size;
            used_count.store(c.used_count);
        }

        Chunk(const Chunk && c){
            //移动构造
            dev_ptr = c.dev_ptr;
            chunk_size = std::move(c.chunk_size);
            class_idx = std::move(c.class_idx);
            allocated_size = std::move(c.allocated_size);
            used_count.store(c.used_count);
        }
    };

    //chunk在集合内部应该按照指针顺序排序，这样允许在O(logn)的时间内直接分配，方便
    struct ChunkSort{
        bool operator()(const Chunk & c1, const Chunk & c2) const {
            return c1.dev_ptr < c2.dev_ptr;
        }
    };

    class SpinLock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
    public:
        void lock(){
            while(flag.test_and_set(std::memory_order_acquire)){
                //TODO:这里应该认为设置这个宏，在cmake那里，这里应该有CPU提供的pause操作，不然比较耗电
                #ifdef __X86_64__
                __asm__ __volatile__("pause" ::: "memory");
                #endif
            }
        }

        void unlock(){
            flag.clear(std::memory_order_release);
            // std::cout << "unlock" << std::endl;
        }
    };

    class GPUMemoryPool::Impl{
        static constexpr size_t kNumSmallClasses = 16;
        static constexpr size_t kSmallSizes[kNumSmallClasses] = {    //最小分配page size为64
            64, 128, 256, 512, 1024, 2048, 4096, 8192,
            16384, 32768, 49152, 65536,
            131072, 262114, 524288, 1048576            
        };

        static constexpr size_t kMaxSmallSize = 1 << 20;
        static constexpr size_t kDefragment_interval = 100;   // 每执行kDefragment_interval次资源释放，进行一次碎片整理

        static constexpr size_t kReleaseThreshold = 256;    //本地释放阈值
        static constexpr size_t kBatchSize = 64; //批量获取的批量大小
        static constexpr size_t kRefillThreshold = kBatchSize * 64;  //中心重新缓存的阈值
        static constexpr size_t kChunkSize = 1 << 21;    //2MB
        static constexpr size_t kMaxGPUSpaceActive = 1 << 32;   //1GB，如果Page Cache的总内存大于1GB了，就要开始考虑释放显存了


        class PageCache {
            //page cache向Center Cache提供交互接口，释放和申请，这在Center Cache中这两步由一个线程完成，因此实际上不用考虑并发性
            std::vector<Chunk> _chunks;
            size_t _allocated_space{0};   
        public:
            ~PageCache(){
                for(auto & chunk : _chunks){
                    free_chunk_to_cuda(chunk);
                }
                _chunks.clear();

                if(_allocated_space > 0){
                    std::cerr << "Do not free all the memory that allocated" << std::endl;
                }
            }

            void allocate_chunks(int size, std::vector<Chunk> & out){
                //申请队列申请一个chunk，这里只有中心Cache会访问，因此不用加锁
                //这里申请size，但是不一定能够给出来size，
                while(size > 0){
                    if(_chunks.size() > 0){
                        Chunk & chunk = _chunks.back(); 
                        //这里就可以把chunk返回了
                        size -= chunk.chunk_size;
                        out.push_back(std::move(chunk));
                        _chunks.pop_back();
                    } else {
                        malloc_chunk_from_cuda(kChunkSize);
                    }
                }
            }


            void deallocate_chunk(const Chunk & chunk){
                _chunks.push_back(std::move(chunk));
                while(_allocated_space > kMaxGPUSpaceActive && !_chunks.empty()){
                    Chunk & chunk_to_free = _chunks.back();
                    free_chunk_to_cuda(chunk_to_free);
                    _chunks.pop_back();
                }
            }

            bool exceedMemory(){
                return _allocated_space > kMaxGPUSpaceActive;
            }

        private:
            void malloc_chunk_from_cuda(size_t size) {
                void* ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, size);
                if (err != cudaSuccess) {
                    std::cout << "cudaMalloc fails" << std::endl;
                }
                _allocated_space += size;
                Chunk new_chunk = {ptr, size, -1, 0, 0, false};
                _chunks.push_back(std::move(new_chunk));
            }

            void free_chunk_to_cuda(Chunk & chunk) {
                void * ptr = chunk.dev_ptr;
                cudaFree(ptr);
                _allocated_space -= chunk.chunk_size;
            }
        };


        class CenterAllocator{
            std::shared_ptr<PageCache> _page_cache; 
            std::vector<moodycamel::ConcurrentQueue<void *>>  _free_list;   //_free_list里可能会有一些失效的指针，因此每次向外分配时，需要检查是否有效
            moodycamel::ConcurrentQueue<AsyncFree> _reclaim_list;   //异步回收队列
            std::vector<std::atomic<int>> _malloc_needs;  //空间分配的缺口，缺多少
            std::set<Chunk, ChunkSort> _chunks;
            std::mutex _chunks_lock; // 使用自旋锁保护_chunks，全场只有这么一个锁
            std::thread _ad_thread;  //该线程负责与Page Cache交互，负责分配和释放
            std::atomic<bool> _destorying{false};
            
            std::condition_variable _livelock_cv;
            std::mutex _livelock;

        public:
            CenterAllocator(std::shared_ptr<PageCache> page_cache):_page_cache(page_cache),\
                _free_list(kNumSmallClasses), _ad_thread(&GPUMemoryPool::Impl::CenterAllocator::async_work, this) ,\
                _malloc_needs(kNumSmallClasses){
                warm_up();
            }

            ~CenterAllocator(){
                _destorying.store(true);
                if(_ad_thread.joinable()) _ad_thread.join();
                cleanup();
            }

            //这里设计单次分配可能并不能得到想要的batch数量，那么就有多少返回多少就行了
            void batch_allocate(int class_idx, size_t batch_size, std::vector<void *>& out){
                if(_malloc_needs[class_idx] < 0){  //如果说以前超发了很多_malloc_needs，这里应该把超发的量补上
                    _malloc_needs[class_idx] += batch_size;
                }

                while(batch_size){
                    void *ptr;
                    if(_free_list[class_idx].try_dequeue(ptr)){
                        std::lock_guard<std::mutex> locker(_chunks_lock);  
                        auto itr = std::upper_bound(_chunks.begin(), _chunks.end(), Chunk{ptr}, ChunkSort());
                        if(itr == _chunks.begin()){
                            std::cout << "unknown logic fault in async_work" << std::endl;
                            break;
                        }
                        --itr;
                        if(static_cast<char*>(itr->dev_ptr) + itr->chunk_size < static_cast<char*>(ptr) || itr->reclaiming){
                            // std::cout << "dev_pre has been free" << std::endl;
                            continue;
                        } 
                        else{
                            itr->used_count++;
                            batch_size--;
                            out.push_back(ptr);
                        }
                    } else break;
                }

                //如果batch_size > 0，那么说明还有一定的缺口，需要让CenterAllocator去异步分配
                if(batch_size > 0){
                    _malloc_needs[class_idx] += batch_size;  //原子操作
                }

                //Debug
                if(out.empty()){
                    //如果一个都没有给出去，那么它要循环到能给出去一个才行
                    void * ptr;
                    while(!_free_list[class_idx].try_dequeue(ptr)){
                        std::unique_lock<std::mutex> locker(_livelock);
                        _livelock_cv.wait(locker, [&]()->bool {return _malloc_needs[class_idx] <= 0;});
                    }  //空转轮询，用尽CPU资源去获取
                    out.push_back(ptr);
                }
            }
            
            void async_deallocate(AsyncFree & af){
                _reclaim_list.enqueue(af);
            }

        private:
            void warm_up(){
                for(int i = 0;i < kNumSmallClasses; ++i){
                    _malloc_needs[i].store(kBatchSize);
                }
            }    

            void async_work(){
                while(!_destorying){
                    //malloc，处理内存缺口
                    for(int i = 0;i < kNumSmallClasses; ++i){
                        if(_malloc_needs[i] > 0){  //有缺口
                            
                            int all_size = _malloc_needs[i] * kSmallSizes[i];
                            int all_chunk_size = (all_size + kChunkSize - 1) / kChunkSize;
                            std::vector<Chunk> out;
                            out.reserve(all_chunk_size);
                            _page_cache->allocate_chunks(_malloc_needs[i] * kSmallSizes[i], out);

                            for(auto & chunk: out) {
                                chunk.class_idx = i;
                                chunk.used_count = 0;
                                chunk.allocated_size = chunk.chunk_size / kSmallSizes[i];
                                _malloc_needs[i] -= chunk.allocated_size;
                                chunk.reclaiming.store(false);
                                {
                                    std::lock_guard<std::mutex> locker(_chunks_lock);
                                    _chunks.insert(std::move(chunk));
                                }

                                //free_list插入
                                for(int j = 0;j < chunk.allocated_size; ++j){
                                    _free_list[i].enqueue(static_cast<char *>(chunk.dev_ptr) + j * kSmallSizes[i]);
                                    _livelock_cv.notify_one();
                                }
                            }
                        }
                    }
                    _livelock_cv.notify_all();  //理论上到这里每一个缺口都已经处理完了，那么就可以通知所有线程了
                    AsyncFree af;
                    int free_count = 256;  //每执行256次回收循环,做一次malloc
                    while(_reclaim_list.try_dequeue(af) && free_count){
                        //这个指针可以回收了，不过回收之前还要考虑两件事情，
                        //1.对应的_malloc_needs有没有缺口，如果有缺口，那么就回收进free_list队列，然后将缺口减1
                        //2.对应的_malloc_needs没有缺口，那就查看page_cache是不是超显存了，如果是的话，考虑在chunk的use_count归0的时候把chunk回收
                        //3.其它情况也直接进free_list
                        
                        //这里只有回收线程会去删除_chunk，因此定位_chunk不用加锁
                        auto chunk = std::upper_bound(_chunks.begin(), _chunks.end(), Chunk{af.ptr}, ChunkSort());
                        if(chunk == _chunks.begin()){
                            std::cout << "unknown logic fault in async_work" << std::endl;
                            break;
                        }
                        --chunk;
                        {
                            std::lock_guard<std::mutex> locker(_chunks_lock);
                            chunk->used_count--;
                            if(chunk->used_count == 0 && _malloc_needs[af.class_idx] == 0 && _page_cache->exceedMemory()){
                                chunk->reclaiming.store(true);  //当chunk标记为true，那么就说明这个chunk已经打算被收回了
                                _chunks.erase(chunk);
                            } else {
                                _free_list[af.class_idx].enqueue(af.ptr);
                            }
                        }
                        free_count--;
                    }
                }
            }

            void cleanup(){
                auto itr = _chunks.begin();
                while(itr != _chunks.end()){
                    _page_cache->deallocate_chunk(*itr);
                    itr = _chunks.erase(itr);
                }
            }

        };
        
        class ThreadCache {  
            std::unordered_map<void *, BlockMeta> _cache_info;
            std::vector<std::vector<void *>> _free_list;
            // moodycamel::ConcurrentQueue<AsyncFree> _deallocate_queue;  //TODO:这里用什么数据结构有待考证
            std::queue<AsyncFree> _deallocate_queue;
            std::shared_ptr<CenterAllocator> _central;
        public:
            ThreadCache(std::shared_ptr<CenterAllocator> & central):_free_list(kNumSmallClasses), _central(central){ }
            ~ThreadCache(){
                auto itr = _cache_info.begin();
                while(itr != _cache_info.end()){
                    //回收应该是异步的
                    if(itr->second.stream != 0) cudaEventRecord(itr->second.ready_event, itr->second.stream);
                    else cudaEventRecord(itr->second.ready_event);
                    
                    _deallocate_queue.push({itr->first, itr->second.class_idx, itr->second.ready_event});
                    _cache_info.erase(itr);
                }

                while(!_deallocate_queue.empty()){
                    AsyncFree af = _deallocate_queue.front();
                    _deallocate_queue.pop();
                    
                    //释放过程
                    if(cudaEventQuery(af.ready_event) == cudaSuccess){
                        cudaEventDestroy(af.ready_event);
                        if(_free_list[af.class_idx].size() <= kReleaseThreshold){
                            _free_list[af.class_idx].push_back(af.ptr);
                        } else {
                            _central->async_deallocate(af);
                        }
                    } else {
                        _deallocate_queue.push(std::move(af));
                    }                    
                }
            }

            void * allocate(size_t size, cudaStream_t stream){
                if(size > kMaxSmallSize){
                    //分配大内存
                }


                int class_idx = get_size_class_index(size);

                //本地缓存有东西就先从本地缓存取东西
                //如果本地没有，从Center_Cache批量获取同类型的Cache
                while(_free_list[class_idx].empty()) refill_local_cache(class_idx);

                void * ptr = _free_list[class_idx].back();
                _free_list[class_idx].pop_back();

                cudaEvent_t event;
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                //记录元数据信息
                _cache_info[ptr] = BlockMeta{size, true, class_idx, stream, event};
                return ptr;
            }

            void * allocate_large(size_t size, cudaStream_t stream){
                void * ptr;
                cudaMalloc(&ptr, size);
                
            }

            void deallocate(void * ptr){
                if(auto itr = _cache_info.find(ptr); itr != _cache_info.end()){
                    //回收应该是异步的
                    if(itr->second.stream != 0) cudaEventRecord(itr->second.ready_event, itr->second.stream);
                    else cudaEventRecord(itr->second.ready_event);
                    
                    _deallocate_queue.push({ptr, itr->second.class_idx, itr->second.ready_event});
                    _cache_info.erase(itr);
                }

                //每次deallocate完了，从队列里面deallocate一些，这里暂定为3
                int deallocate_items = 3;
                while(deallocate_items--){
                    if(!_deallocate_queue.empty()){
                        AsyncFree af = _deallocate_queue.front();
                        _deallocate_queue.pop();
                        
                        //释放过程
                        if(cudaEventQuery(af.ready_event) == cudaSuccess){
                            cudaEventDestroy(af.ready_event);
                            if(_free_list[af.class_idx].size() <= kReleaseThreshold){
                                _free_list[af.class_idx].push_back(ptr);
                            } else {
                                _central->async_deallocate(af);
                            }
                        } else {
                            _deallocate_queue.push(std::move(af));
                        }

                    } else break;
                }
            }
        
        private:
            void refill_local_cache(int class_idx){
                std::vector<void*> new_blocks;
                new_blocks.reserve(kBatchSize);
                
                _central->batch_allocate(class_idx, kBatchSize, new_blocks);  
                
                // 加入本地空闲列表
                _free_list[class_idx].insert(
                    _free_list[class_idx].end(),
                    new_blocks.begin(),
                    new_blocks.end()
                );
            }            
        };

        std::shared_ptr<PageCache> pagecache_;
        std::shared_ptr<CenterAllocator> central_;

    public:
        Impl(){
            pagecache_ = std::make_shared<PageCache>();
            central_ = std::make_shared<CenterAllocator>(pagecache_);
        }

        ~Impl(){}

    private:
        static int get_size_class_index(size_t block_size){
                // 处理超过最大小内存的情况
            if (block_size > kMaxSmallSize) 
                return kNumSmallClasses; // 返回无效索引（调用方需处理）

            int l = 0, r = kNumSmallClasses - 1, mid = 0; //二分查找寻找对应的index
            // 搜索第一个 >= block_size 的规格
            while(l <= r){
                mid = l + (r - l) / 2;
                if(kSmallSizes[mid] < block_size) l = mid + 1;
                else r = mid - 1;
            }

            return l; // fallback
        }

        ThreadCache & allocator(){
            thread_local static ThreadCache allocator(central_);
            return allocator;
        }
    
    public:
        void * allocate(size_t size, cudaStream_t stream){
            return allocator().allocate(size, stream);
        }

        void deallocate(void * ptr){
            if(!ptr) return;
            allocator().deallocate(ptr);
        }

        void cleanup(){

        }
    };

    // thread_local GPUMemoryPool::Impl::ThreadCache GPUMemoryPool::Impl::_thread_local_cache;

    GPUMemoryPool::GPUMemoryPool() : _impl(std::make_unique<Impl>()){}

    GPUMemoryPool::~GPUMemoryPool(){
        cleanup();
    }

    GPUMemoryPool& GPUMemoryPool::instance() {
        static GPUMemoryPool instance;  //单例模式
        return instance;
    }

    void * GPUMemoryPool::allocate(size_t size, cudaStream_t stream){
        return _impl->allocate(size, stream);
    }
    
    void GPUMemoryPool::deallocate(void * ptr){
        _impl->deallocate(ptr);
    }

    void GPUMemoryPool::cleanup(){
        _impl->cleanup(); 
    }
}