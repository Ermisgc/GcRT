/**
 * @file manager端，可以把引擎当作中间件，支持管理员外部调整和跟踪
 * @class ModelManager 封装有关管理员的身份信息，包括其本身的管理员id, 管理员向外提供的私钥，自己本身可以携带的公钥信息等
 * @class ManageMessage 管理员可以基于二进制协议/自研的Rpc协议向服务器发起消息,这里的消息包括具体的指令、可能携带的新增模型文件
 * @class ManageRequest 包装管理端对Server的管理信息、也能解析出具体的管理信息
 * @class ManageResponse 包装Server端对管理端的响应
 * @class TCPClient 封装好的TCPClient
 * @details 管理端的调用过程：
 * ModelManager + ManageMessage  --- ManageRequest ---> TCPClient ---> [Server]集群
 * ----- Manager Wait -----
 * [Server]集群处理  --->  返回实时进度...  ---> 处理完成   ---> TCPClient
 * TCPClient  ---> ManagerResponse  ---> manager
*/
