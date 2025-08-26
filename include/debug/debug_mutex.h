#include <mutex>
#include <condition_variable>
#include "debug_logger.h"

#define varName(x) #x

namespace GcRT{
namespace debug{
    class DebugMutex {
    public:
        DebugMutex(const std::string & name = "Unnamed") : _name(name) {}
        
        inline void lock() {
            DBG_LOG(_name + ": wait to lock");
            mutex.lock();
            DBG_LOG(_name + ": acquire lock");
        }
        
        inline void unlock() {
            mutex.unlock();
            DBG_LOG(_name + ": unlock");
        }
        
        bool try_lock() {
            bool result = mutex.try_lock();
            return result;
        }
        
    private:    
        std::string _name;
        std::mutex mutex;  // 实际的互斥量
    };


    class DebugConditionVariable {
    public:
        DebugConditionVariable(const std::string name = "UnnamedCV") : _name(name) {}
        
        inline void notify_one() noexcept {
            DBG_LOG(_name + ": Notify 1");
            cv.notify_one();
        }
        
        inline void notify_all() noexcept {
            DBG_LOG(_name + ": Notify All");
            cv.notify_all();
        }
        
        template <typename Lock>
        inline void wait(Lock& lock) {
            DBG_LOG(_name + ": Wait");
            cv.wait(lock);
            DBG_LOG(_name + ": Awake");
        }
        
        template <typename Lock, typename Predicate>
        inline void wait(Lock& lock, Predicate pred) {
            DBG_LOG(_name + ": Wait");
            cv.wait(lock, pred);
            DBG_LOG(_name + ": Awake");
        }
        
        template <typename Lock, typename Rep, typename Period>
        inline std::cv_status wait_for(Lock& lock, const std::chrono::duration<Rep, Period>& rel_time) {
            DBG_LOG(_name + ": Wait For");
            auto status = cv.wait_for(lock, rel_time);
            DBG_LOG(_name + ": Awake");
            return status;
        }
        
        template <typename Lock, typename Rep, typename Period, typename Predicate>
        inline bool wait_for(Lock& lock, const std::chrono::duration<Rep, Period>& rel_time, Predicate pred) {
            DBG_LOG(_name + ": Wait For");
            bool result = cv.wait_for(lock, rel_time, pred);
            DBG_LOG(_name + ": Awake");
            return result;
        }
        
        template <typename Lock, typename Clock, typename Duration>
        std::cv_status wait_until(Lock& lock, const std::chrono::time_point<Clock, Duration>& timeout_time) {
            log_wait_until();
            auto status = cv.wait_until(lock, timeout_time);
            log_awake();
            return status;
        }
        
        template <typename Lock, typename Clock, typename Duration, typename Predicate>
        bool wait_until(Lock& lock, const std::chrono::time_point<Clock, Duration>& timeout_time, Predicate pred) {
            log_wait_until();
            bool result = cv.wait_until(lock, timeout_time, pred);
            log_awake();
            return result;
        }
        
        const std::string _name;
        std::condition_variable_any cv;  // 可以与任何锁类型配合使用
    };
} 

    // 定义锁守卫类型
    using namespace debug;
    using DebugLockGuard = std::lock_guard<DebugMutex>;
    using DebugUniqueLock = std::unique_lock<DebugMutex>;
    
}