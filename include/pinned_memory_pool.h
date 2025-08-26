#ifndef PINNED_MEMORY_POOL_H
#define PINNED_MEMORY_POOL_H
/**
 * PinnedMemoryPool，其特点是存在于CPU中的Pinned Memory
 * 当接收到数据请求，应该从PinnedMemoryPool里面获取一个数据指针，然后将数据拷贝到PinnedMemoryPool的数据指针中
 * 这样当执行CudaMemcpy将数据从PinnedMemoryPool拷贝到显存池时，就可以省去从Pageable内存拷贝到固定内存的一个步骤，在数据量大时提高效率
 */

#include <NvInfer.h>
#include <optional>

namespace GcRT{
    namespace inner{
        class PinnedMemoryPool{
            void * device_ptr;
        public:
            PinnedMemoryPool() = delete;
            PinnedMemoryPool(size_t bytes);
            ~PinnedMemoryPool();

            /**
             * @brief 分配一个大小为bytes的内存
             */
            std::optional<void *> Allocate(size_t bytes);
        };
    }
}
#endif