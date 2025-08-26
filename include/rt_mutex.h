#include "debug/debug_mutex.h"

namespace GcRT{
#ifndef DEBUG
    // 使用DebugMutex作为默认互斥量
    using Mutex = DebugMutex;
    using LockGuard = DebugLockGuard;
    using UniqueLock = DebugUniqueLock;
    using ConditionVariable = DebugConditionVariable;
#else
    using Mutex = std::mutex;
    using LockGuard = std::lock_guard<std::mutex>;
    using UniqueLock = std::unique_lock<std::mutex>;
    using ConditionVariable = std::condition_variable;
#endif


}
