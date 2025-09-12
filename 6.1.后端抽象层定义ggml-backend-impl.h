/**
 *   在这个 文件中 定义了 大量的 结构体 
 *   结构体 的字段 定义了 关键 属性和 函数指针
 * 
 *  有一个 全局表， 表中有所有的后端注册描述符  和 所有的设备实例
 *
- 注册表(ggml_backend_reg) ：是“某一个后端（如 CPU/CUDA/CANN）的注册描述符
- 设备(ggml_backend_dev) ：表示具体的计算设备（如CPU、GPU等）
- 后端(ggml_backend) ：后端实例  内部封装一个或多个实际底层流/队列 承载：提交图、同步、异步拷贝
- 缓冲区类型(ggml_backend_buffer_type) ：缓冲区的工厂类，定义如何创建和管理缓冲区
- 缓冲区(ggml_backend_buffer) ：实际存储数据的内存区域
- 上下文(context) ：各组件内部使用的私有数据指针
- 后端事件(ggml_backend_event) ：用于不同后端流之间的同步机制

 */

 // 后端注册描述符
struct ggml_backend_reg {
    int api_version;                          // API版本号
    struct ggml_backend_reg_i iface;          // 注册描述符函数接口函数表        
    void * context;                           // 上下文指针
};

// 内部后端注册API
GGML_API void ggml_backend_register(ggml_backend_reg_t reg);


// 设备
struct ggml_backend_device {
    struct ggml_backend_device_i iface;
                *get_name              // 获取设备名称
                *get_description       // 获取设备描述
                *get_memory            // 获取设备内存
                *get_type              // 获取设备类型
                *get_props             // 获取设备属性
                *init_backend          // 初始化后端              //后端的初始化 在 设备的接口中
                *get_buffer_type       // 获取缓冲区类型
                *get_host_buffer_type  // 获取主机缓冲区类型
                *buffer_from_host_ptr  // 从主机指针创建缓冲区
                *supports_op           // 支持操作
                *supports_buft         // 支持缓冲区类型
                *offload_op            // 卸载算子
                *event_new             // 创建事件
                *event_free            // 释放事件
                *event_synchronize     // 同步事件
    ggml_backend_reg_t reg;
    void * context;
};



// 后端
struct ggml_backend {
    ggml_guid_t guid;                         // 唯一标识符
    struct ggml_backend_i iface;              // 后端接口函数表
                *get_name    // 获取后端名称
                *free        // 释放后端
                *set_tensor_async   // 设置异步张量
                *get_tensor_async   // 获取异步张量
                *cpy_tensor_async   // 异步拷贝张量
                *synchronize        // 同步后端
                *graph_plan_create  // 创建计算图执行计划
                *graph_plan_free    // 释放计算图执行计划
                *graph_plan_update  // 更新计算图执行计划
                *graph_plan_compute // 执行计算图计划
                *graph_compute      // 直接执行计算图
                *event_record       // 记录事件
                *event_wait         // 等待事件
    ggml_backend_dev device;                    // 所属设备
    void * context;                             // 上下文指针
};


// 后端缓冲区
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;       // 缓冲区接口函数表    
                *free_buffer    // 释放缓冲区
                *get_base       // 获取缓冲区基础地址
                *init_tensor    // 初始化张量
                *memset_tensor  //  memset 张量
                *set_tensor     // 设置张量
                *get_tensor     // 获取张量
                *cpy_tensor     // 拷贝张量
                *reset          // 重置缓冲区
    ggml_backend_buffer_type buft;            // 缓冲区类型
    void * context;                           // 上下文指针
    size_t size;                              // 缓冲区大小
};

// 后端事件
struct ggml_backend_event {
    struct ggml_backend_device * device;      // 关联设备
    void * context;                           // 上下文指针
};


// 后端调度器，这个结构体也很重要，只是源码中是放在 ggml-backend.cpp中
struct ggml_backend_sched {
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    struct ggml_cgraph graph;
    // graph splits
    struct ggml_backend_sched_split * splits;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_context * ctx;
};




/* 这些结构体的所属关系

 a. 注册表 ↔ 设备  一对多   一个注册表可以管理多个设备实例   可以通过注册表 索引访问设备   全局还有一个“聚合注册表”保存多个 backend_reg。

 b. 设备 ↔ 后端  可以一对多，通常一对一   可以从同一设备多次调用 ggml_backend_dev_init() 生成多个 ggml_backend。常规使用里很少创建多个，但设计允许。

 c. 设备 ↔ 缓冲区类型  一对一   一个设备提供一个缓冲区 一整块大内存

 f. 上下文(context)与各组件  一对一  每个组件实例都有自己的私有上下文指针   可以通过上下文 访问组件内部数据

 g. 后端事件与设备  多对一   一个设备可以创建多个事件用于流同步   可以通过事件 查到他所属设备

*/


/*  注册表  设备类型描述   设备实例   后端实例  概念区分

全局聚合注册表 (static ggml_backend_registry)	
    全局单例对象，内部持有多个 设备类型描述(reg)  CUDA-reg、CANN-reg、CPU-reg  每一个描述了一种设备

设备类型描述 (ggml_backend_reg)	
    一个reg对象，描述了一种设备，可以通过这个对象 扫描出有个几个显卡

设备对象 (ggml_backend_dev)	
    每个设备对象 就表示一张卡	通过该对象可以 init 出过个会话 backend

后端实例 (ggml_backend)	
    上面 init 出来的 backend 实例，
    在某个设备上建立的执行上下文	
    内部可含具体 stream/queue	llama_context 用于执行图

*/



/* 关键调用流程

1. 程序第一次访问 backend 枚举接口 → 构造全局 registry，这里就是实例化出了全局聚合注册表 对象
2. 在实例化全局聚合注册表 对象的时候，在其构造函数中就会调用 register_backend(ggml_backend_cann_reg())：
        在ggml_backend_cann_reg()中就会 实例化出 cann 的 设备类型描述对象 ggml_backend_reg
        通过 设备类型描述对象 就可以 初始化 ACL (aclInit) 扫描出 有几张卡 
        为每张卡创建一个 ggml_backend_device 都登记到 设备类型描述对象 中
4. register_backend() 把 reg 加到 backends，并循环把 CANN0、CANN1 等放入全局 devices。
    ========gglm的接口就这些，下面就是llama.cpp怎么使用了=======
5.模型加载后决定用哪些设备 → llama_context 中调用：
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    生成真正执行用 backend 实例。   

*/
