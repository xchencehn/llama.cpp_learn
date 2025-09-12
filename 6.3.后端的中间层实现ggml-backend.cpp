
/*       这个文件 全是函数的实现 使用函数名 来分类
           这里的函数 有一些是 在 ggml-backend.h 中定义好的 上层交互接口， 
                    有一些是 在 ggml-backend-impl.h 中定义好的 哪些结构体中的函数指针接口，
                    有一些是为了实现接口又定义的辅助函数


          这里前面的一些 函数都是 对 设备实例  后端实例 后端缓冲实例 的 set get 类函数
          后端的 一大段 是对 计算图要如何在 多个GPU 以及 GPU+CPU上计算，做了实现，这部分比较关键
          
          设计调度器的目的是为了随心所欲的控制ggml能最大化的利用硬件资源来执行计算图：
            1. 计算图只需要定义计算逻辑就可以了，它以张量为节点，算子为边，构成了一个有向无环图，使用 cgraph表示
            2. 调度器中定义 这样计算图中的哪些张量要放在cpu主存上，哪些张量要放在设备上，
                哪些算子要用cpu后端算，哪些算子要用设备算，以及多个设备，那个设备算哪个，都做了详细的规划
                启动计算后，各个后端就无脑执行分给它的计算任务就行了。
            3. 调度器提供了钩子机制，可以让llama.cpp设置一些函数，让该函数在某个算子完成或者失败后触发执行
                这样就能实现大量的过程监控和旁路输出操作。
*/

//  这7个 输入的是 后端缓冲区类型
ggml_backend_buft_name(ggml_backend_buffer_type_t buft) // 获取后端缓冲区类型的名称
ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) // 分配后端缓冲区
ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) // 获取后端缓冲区类型的对齐要求
ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) // 获取后端缓冲区类型的最大大小
ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) // 获取后端缓冲区类型的分配大小
ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) // 判断后端缓冲区类型是否为主机缓冲区
ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) // 获取后端缓冲区类型的设备


//  这16个 输入的是 后端缓冲区实例， 操作缓冲区
ggml_backend_buffer_init(ggml_backend_buffer_type_t buft, struct ggml_backend_buffer_iface iface, void * context, size_t size) // 初始化后端缓冲区
ggml_backend_buffer_name(ggml_backend_buffer_t buffer) // 获取后端缓冲区的名称
ggml_backend_buffer_free(ggml_backend_buffer_t buffer)  // 释放后端缓冲区
ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer)  // 获取后端缓冲区的大小
ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer)  // 获取后端缓冲区的基础指针
ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) // 初始化后端缓冲区的张量
ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value)  // 清除后端缓冲区的内容
ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer)   // 获取后端缓冲区的对齐要求
ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer)   // 获取后端缓冲区的最大大小
ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor)  // 获取后端缓冲区的分配大小
ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) // 判断后端缓冲区是否为主机缓冲区
ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage)  // 设置后端缓冲区的使用方式
ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer)  // 获取后端缓冲区的使用方式
ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer)  // 获取后端缓冲区的类型
ggml_backend_buffer_reset(ggml_backend_buffer_t buffer)   // 重置后端缓冲区
ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst)   // 复制后端缓冲区的张量


// 这里 21 个函数 输入的就是 具体后端的实例， 直接对真实后端操作
ggml_backend_guid(ggml_backend_t backend)   // 获取后端的唯一标识符
ggml_backend_name(ggml_backend_t backend)   // 获取后端的名称
ggml_backend_free(ggml_backend_t backend)   // 释放后端
ggml_backend_get_default_buffer_type(ggml_backend_t backend)   // 获取后端的默认缓冲区类型
ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size)   // 分配后端缓冲区
ggml_backend_get_alignment(ggml_backend_t backend)    // 获取后端的对齐要求
ggml_backend_get_max_size(ggml_backend_t backend)    // 获取后端的最大大小
ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size)    // 异步设置后端张量的内容
ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size)    // 异步获取后端张量的内容
ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size)    // 设置后端张量的内容
ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size)    // 获取后端张量的内容
ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size)    // 异步设置后端张量的内容
ggml_backend_synchronize(ggml_backend_t backend)    // 同步后端
ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph)    // 创建后端计算图计划
ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan)   // 释放后端计算图计划
ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan)   // 执行后端计算图计划
ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph)   // 执行后端计算图
ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph)    // 异步执行后端计算图
ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op)    // 判断后端是否支持计算图中的操作
ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op)    // 后端是否支持将操作卸载到设备
ggml_backend_get_device(ggml_backend_t backend)    // 获取后端的设备

// 这两个是 单独对 张量 拷贝用的
ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst)    // 复制后端张量的内容
ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst)    // 异步复制后端张量的内容

//  这里就是 对 后端事件 操作的函数，  后端事件 就是 多流同步 和 异步控制的时候使用的
ggml_backend_event_new(ggml_backend_dev_t device)    // 创建后端事件
ggml_backend_event_free(ggml_backend_event_t event)    // 释放后端事件
ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend)  // 记录事件
ggml_backend_event_synchronize(ggml_backend_event_t event)  // 同步事件
ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event)  // 等待事件


// 这14 个 直接对 设备对象 操作
ggml_backend_dev_name(ggml_backend_dev_t device)  //返回设备名称
ggml_backend_dev_description(ggml_backend_dev_t device)    //返回设备描述
ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total)  //返回设备内存信息
ggml_backend_dev_type(ggml_backend_dev_t device)    //返回设备类型
ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props)    //返回设备属性
ggml_backend_dev_backend_reg(ggml_backend_dev_t device)  //返回设备后端注册信息
ggml_backend_dev_init(ggml_backend_dev_t device, const char * params)  //初始化后端
ggml_backend_dev_buffer_type(ggml_backend_dev_t device)    //返回设备缓冲区类型
ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device)    //返回设备主机缓冲区类型
ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size)  // 从主机指针创建后端缓冲区
ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op)   // 判断设备是否支持计算图中的操作
ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft)   // 判断设备是否支持缓冲区类型
ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op)   // 判断设备是否支持将操作卸载到设备

// 这里是对 后端注册描述符 操作   当时费劲定义那个结构体，就是为了链接这里的函数的
ggml_backend_reg_name(ggml_backend_reg_t reg)   // 获取后端注册描述符的名称
ggml_backend_reg_dev_count(ggml_backend_reg_t reg)    // 获取后端注册描述符的设备数量
ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index)   // 获取后端注册描述符的设备
ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name)    // 获取后端注册描述符的过程地址


/*
// scheduler// 接下来就 ggmll后端的调度功能的实现了
        // 这里的是 计算图执行的 一部分， 中间层， 它对上 提供了 计算图的 同步和异步两个执行接口，
        // 要使用这个接口，需要传入两个参数， 1. 调度器， 2.计算图
        // 调度器就是这个结构器的实例，这个结构体的使用 6 个字段说清楚了一个计算图应该怎么执行。

            1. 资源清单字段
            int n_backends   有多少个可用的硬件后端。
            ggml_backend_t backends[16]    后端列表，比如数组第一位 backends[0] 是 CPU，后面的8位是 CUDA 
            ggml_backend_buffer_type_t bufts[16]    和上一个字段一一对应，描述每个后端对应的buffer类型。

            2. 执行计划字段
            int * node_backend_ids    一个数组，大小与计算图节点数相同。 
                                      数组的每个元素，对应计算图的一个节点（张量），这个元素描述了该节点会被分配到哪个后端上计算，实际存的就是后端id
            
            struct ggml_backend_sched_split * splits 光靠上面的数组和哈希表来构建整个调度系统，过于稀碎，再次基础上
                                                     我们应该使用子图来描述执行计划，一个完整的计算图cgraph, 如果都在一个设备上执行，那么就不用写切分
                                                     但是如果要在多个设备上执行，就要切分成多个子图， 例如显存不够，一部分在cpu上，一部分在gpu上， 或者例如80层的模型8卡流水线执行，每张卡10层
                                                     这个字段 splits 就是来存储切开后的所有子图的，这个一个动态数组，切成几份，就存几张子图，每个元素就是一个子图。
                                                     子图的数据结构如下：
                                                        struct ggml_backend_sched_split{
                                                                backend_id   表示这个子图由哪个后端执行  这个数字和 之前的数组 node_backend_ids 中的值是能对上的
                                                                i_start, i_end  表示这个子图在原始大图中的节点起止范围 
                                                                struct ggml_cgraph graph  子图本身真正的数据结构也是一个 cgraph
                                                        }
                                                        
            
            3. 同步工具字段
            ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES]
                    用于后端之间同步的“事件”对象。比如，流水线执行的时候  我们就可以用这个二维数组来控制多个设备的同步
                    第一维表示那个设备，第二维表示流水线并行度
                    例如 8 卡 直推一个序列， 那就是 events[8][1]   不考虑batch
                    又如 8 卡 要 4 个序列并行流水  那就是 events[8][4]  
                        这样 0号卡 处理好第一个序列的 第一个token 就触发 events[0][0]  1号卡就知道了，开始拉取第一个序列第一个token隐藏状态,接着计算， 计算完就触发 events[1][0] , 3号卡完了会触发events[2][0]
                            0号卡 立马开始处理第二个序列的 第一个token, 处理完就触发 events[0][1], 2号就会收到消息，开始处理，处理完了触发 events[1][1]
                            0号卡 立马开始处理第三个序列的 第一个token, 处理完就触发 events[0][2],
                            0号卡 立马开始处理第四个序列的 第一个token, 处理完就触发 events[0][3],
                            0号卡  一看 第一个序列的第一个token都采样出来了，立马开始第一个序列的第二个token计算， 完成就触发 events[0][0]
                            ...
                    
                    
        
*/
struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;
};

ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) // 获取调度器中指定后端的ID
ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) // 从缓冲区获取张量操作对应的后端
ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) // 从当前张量获取对应的后端ID
ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) // 打印调度器中计算图节点的后端分配信息
ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) // 判断指定后端是否支持张量的缓冲区
ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) // 对计算图进行切分，以便在不同后端执行
ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) // 为调度器分配子图切分所需的内存
ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) // 计算调度器中子图的执行计划
ggml_backend_sched_new(ggml_backend_t * backends,ggml_backend_buffer_type_t * bufts,int n_backends,size_t graph_size,bool parallel,bool op_offload) // 创建一个新的调度器实例
ggml_backend_sched_free(ggml_backend_sched_t sched) // 释放调度器占用的资源
ggml_backend_sched_reset(ggml_backend_sched_t sched) // 重置调度器状态
ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) // 为调度器预留资源
ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) // 为计算图分配调度器所需的资源
ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) // 同步执行调度器中的计算图
ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) // 异步执行调度器中的计算图
ggml_backend_sched_synchronize(ggml_backend_sched_t sched) // 同步调度器中的操作
ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) // 设置调度器的评估回调函数
ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) // 获取调度器中子图的数量
ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) // 获取调度器中流水线并行的副本数量
ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) // 获取调度器中可用的后端数量
ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) // 获取调度器中指定索引的后端
ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) // 获取指定后端的缓冲区大小
ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) // 设置张量对应的后端
ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) // 获取张量对应的后端
ggml_backend_view_init(struct ggml_tensor * tensor)  // 初始化后端视图
ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr)// 为指定的张量在给定的后端缓冲区中分配内存
graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) // 复制计算图中的张量
graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) // 初始化计算图中节点的副本
ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) //复制后端计算图
ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) // 释放复制的后端计算图所占用的资源
ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, 
    ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor * test_node) // 比较两个后端在给定计算图上的执行结果




//  下面就是 和 设备后端buffer一样的 东西 在cpu侧又实现了一遍， cpu后端是必须的，某个算子设备后端不支持了，就会下发到cpu后端上执行
// CPU backend - buffer
ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer)
ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer)
ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size)
ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size)
ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size)
ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst)
ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value)
static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft)
ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size)
ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft)
ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft)
ggml_backend_cpu_buffer_type(void)
ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft)
ggml_backend_cpu_buffer_from_ptr_type(void)
ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size)