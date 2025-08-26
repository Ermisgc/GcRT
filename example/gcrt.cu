/**
 * @file gcrt.cu 这里应该实现推理器、接收两端请求：来自用户侧和来自管理侧的请求
 * @section Manager
 * @class Manager 它是一个Rpc服务,它能提供模型的传输与接收、模型的列表、热启动/加载模型
 * @section Core
 * @class GcRT 它是核心部分，提供四个部分的功能：1. 信息的收发器，来自客户端和管理端的信息处理（主线程 + 若干poller）
 * 2. 显存池GPUAllocator和内存池PinnedMemoryPool，GcRT本身会做信息的收发，信息的收发中，数据部分将首先拷贝到Pinned Memory池，然后再由显存池分配显存，执行拷贝
 * 3. 推理器Inferencer，执行推理，中间也包含内存调度，应该要支持多线程
 * 4. 结果处理器ResultProcesser，由于是异步的推理框架，推理完成后，CUDA流将调用对应的异步函数将结果交给ResultProcesser进行进一步处理
 */