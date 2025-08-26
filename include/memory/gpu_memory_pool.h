#include <NvInfer.h>
#include <memory>

namespace GcRT{
    class GPUMemoryPool{
    public:
        static GPUMemoryPool & instance();

        GPUMemoryPool(const GPUMemoryPool &) = delete;
        void operator=(const GPUMemoryPool &) = delete;

        void * allocate(size_t size, cudaStream_t stream = 0);
        void deallocate(void * ptr);

        void cleanup();
    
    private:
        GPUMemoryPool();
        ~GPUMemoryPool();

        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}