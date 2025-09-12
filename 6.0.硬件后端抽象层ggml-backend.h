/**
 * 
 首先 ggml-backend.h 是上层代码与后端交互的唯一入口 ， 具体操作接口都会 汇总到这个文件中。

 具体后端的抽象设计都分布在如下三个文件中，
ggml-backend-impl.h:    定义了所有硬件后端（CPU, CUDA, Metal等）必须遵守的接口规范。
ggml-backend.cpp:       实现了调用具体后端接口的通用逻辑和计算任务调度器，能将一个计算图拆分到多个硬件上执行。
ggml-backend-reg.cpp:   负责管理、注册和发现可用的硬件后端，支持静态编译和动态加载。

这里的抽象设计指的是做了如下3件事情：
    1. 定义了具体的后端 例如  cann  cuda 等 应该怎么实现，给出了 结构体，结构体中包含了必须的字段和需要实现的函数指针   ggml-backend-impl.h
    2. 实现了 后端类型发现，通过后端类型找到设备， 将设备注册到 全局后端管理表中   backend-reg.cpp 
    3. 实现了 统一的调用逻辑， 具体的后端只需要实现具体的功能函数即可， 至于这些 函数怎么调用，已经在框架层实现了 backend.cpp
 */


 //  ggml-backend.h 中汇总的 接口如下：

 //              这些都是 从 ggml-backend-impl.h  ggml-backend.cpp  ggml-backend-reg.cpp 中汇总而来，供上层代码调用




/*

1. 既然这些接口都是上层应用交互的 那上层应用要和我们交互些什么东西呢？
    1. 上层应用 就是 llama.cpp,  llama.cpp 要和 ggml 交互， ggml 作为 llama.cpp的引擎
    2. llama.cpp 要 通过 一系列的接口让 ggml 顺利的跑起来，能干活，还能控制。
            1.要知道编译的时候 支持了什么类型的后端，
            2.要知道 这种后端 有几张卡
            3.要给出初始化接口，允许llama.cpp 通过传参控制初始化的过程
            4.初始化完成后，要给出接口让 llama.cpp 知道可用的显存大小，当前状态等信息
            5.要给出接口让llama.cpp可以控制ggml的buffer怎么分配
            6.还要给出接口让llama.cpp可以随意的读写刚才分配的buffer，方便加载权重，和读取结果
            7.权重加载进去后，llama.cpp 就会自己组装好计算图，然后让ggml来执行
            8.ggml要给出 同步执行的接口， 异步执行的接口，不同调度策略的接口等，让llama.cpp尽可能随心所欲的驾驭ggml
            9.同时为了监控 ggml 内部究竟在发生什么，以及为了实现 当ggml处于某时刻的时候，要出发llama.cpp做什么事情
            10.这就需要ggml有事件机制，给出 插入事件和等待事件的接口

基于以上需求，设计了如下接口，来驱使ggml
*/

/*
上层应用与后端系统的典型交互流程如下：
    1.发现设备
    2.初始化
    3.资源准备
    4.计算执行
    5.同步与数据获取
    6.资源释放
*/


// 核心类型定义
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_event * ggml_backend_event_t;
typedef struct ggml_backend * ggml_backend_t;
typedef void * ggml_backend_graph_plan_t;
typedef struct ggml_backend_reg * ggml_backend_reg_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;

// 设备类型枚举
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,    // CPU 设备
    GGML_BACKEND_DEVICE_TYPE_GPU,    // GPU 设备
    GGML_BACKEND_DEVICE_TYPE_ACCEL   // 加速器设备
};

// 设备功能特性
struct ggml_backend_dev_caps {
    bool async;              // 异步操作支持
    bool host_buffer;        // 固定主机缓冲区支持
    bool buffer_from_host_ptr; // 从主机指针创建缓冲区
    bool events;             // 事件同步支持
};

// 设备属性
struct ggml_backend_dev_props {
    const char * name;        // 设备名称
    const char * description; // 设备描述
    size_t memory_free;       // 可用内存
    size_t memory_total;      // 总内存
    enum ggml_backend_dev_type type; // 设备类型
    struct ggml_backend_dev_caps caps; // 设备功能
};

// 缓冲区使用类型
enum ggml_backend_buffer_usage {
    GGML_BACKEND_BUFFER_USAGE_ANY = 0,     // 通用用途
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1, // 模型权重
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2  // 计算缓冲区
};

// 后端特征
struct ggml_backend_feature {
    const char * name;  // 特征名称
    const char * value; // 特征值
};


/*
   这些接口都会在 ggml-backend-imp.h 中的结构体在实例化的时候，
   填入对应的 iface 字段完成绑定，
   调用的时候都是 通过 ggml_backend->iface.init_backend(device, params) 来调用的
*/


// ================ 后端注册表管理 API ================
GGML_API size_t             ggml_backend_reg_count(void);
GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);
GGML_API size_t             ggml_backend_dev_count(void);
GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);


// ================ 后端注册表使用 API ================
GGML_API const char *       ggml_backend_reg_name(ggml_backend_reg_t reg);
GGML_API size_t             ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
GGML_API void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);


// ================ 后端设备 API ================
GGML_API const char *                  ggml_backend_dev_name(ggml_backend_dev_t device);
GGML_API const char *                  ggml_backend_dev_description(ggml_backend_dev_t device);
GGML_API void                          ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
GGML_API enum ggml_backend_dev_type    ggml_backend_dev_type(ggml_backend_dev_t device);
GGML_API void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
GGML_API ggml_backend_reg_t            ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
GGML_API ggml_backend_t                ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
GGML_API bool                          ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);


// ================ 后端API ================
GGML_API const char * ggml_backend_name(ggml_backend_t backend);
GGML_API void         ggml_backend_free(ggml_backend_t backend);
GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
GGML_API void             ggml_backend_synchronize(ggml_backend_t backend);
GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);


// ================ 后端初始化便捷函数 ================
GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
GGML_API ggml_backend_t ggml_backend_init_best(void);
GGML_API void           ggml_backend_load_all(void);


// ================ 后端缓冲区类型 API ================
GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);


// ================ 后端缓冲区 API ================
GGML_API const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
GGML_API void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
GGML_API void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
GGML_API size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
GGML_API enum ggml_status               ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
GGML_API void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
GGML_API ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
 

// ================ 后端调度器 API ================
typedef struct ggml_backend_sched * ggml_backend_sched_t;
GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);


// ================ 事件 API ================
GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
GGML_API void                 ggml_backend_event_free(ggml_backend_event_t event);
GGML_API void                 ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
GGML_API void                 ggml_backend_event_synchronize(ggml_backend_event_t event);


// ================ CPU 后端便捷函数 ================
GGML_API ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);



 

