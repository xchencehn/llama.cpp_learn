
/*  
          设计调度器的目的是为了随心所欲的控制ggml能最大化的利用硬件资源来执行计算图：
            1. 计算图只需要定义计算逻辑就可以了，它以张量为节点，算子为边，构成了一个有向无环图，使用 cgraph表示
            2. 调度器中定义 这样计算图中的哪些张量要放在cpu主存上，哪些张量要放在设备上，
                哪些算子要用cpu后端算，哪些算子要用设备算，以及多个设备，那个设备算哪个，都做了详细的规划
                启动计算后，各个后端就无脑执行分给它的计算任务就行了。
            3. 调度器提供了钩子机制，可以让llama.cpp设置一些函数，让该函数在某个算子完成或者失败后触发执行
                这样就能实现大量的过程监控和旁路输出操作。
*/

/* 调度器的初始化过程


    1.执行命令 ./build/bin/llama-cli -m /home/chenchen/models/gemma-3-4b-it-f16.gguf -p '用一句话介绍你自己。' -no-cnv -ngl 100 -sm none -mg 0 
    2.就会从这里的 main 函数开始执行
    3.   依次执行了   common_init();   // 只是初始化了日志工具，时钟对齐
                    common_init_from_params(params); 这里开始真正的初始化
                                model = llama_model_load_from_file(params.model.path.c_str(), mparams);
                                cparams = common_context_params_to_llama(params);
                                lctx = llama_init_from_model(model, cparams);
                                            ctx = new llama_context(*model, params);   // 这里开始初始化上下文，在llama_context的构造方法中完成了大量的初始化工作
                                                            1.初始化了 GPU后端，加速卡后端，CPU后端，线程池等
                                                            2.然后，类似的调用流程 初始化了  CPU backend
                                                            3.到此为止，有了 9 个后端，然后把这 9 个后端的 名为ggml_backend_set_n_threads的函数接口到 放到一个里表中，方便后面给每个后端设置线程数
                                                            4.设置 abort 回调  这里埋入了一个中断函数指针，该函数指针在推理时每个node结束都会扫一眼，如果有中断信号，就该图退出推理。
                                                            8. 分配输出缓冲区（logits/embd）    调用  output_reserve  分配 logits/embedding 缓冲区   这是构造过程中第一处大内存分配  输出 buffer 的大小 ≈  n_seq_max * (n_vocab + n_embd) * sizeof(float) 
                                                            9. 初始化 Memory 模块（KV cache）    根据参数分配 KV cache（存 K/V 张量用来复用注意力）  会预分配 KV tensor 的 backend buffer
                     ################################      10. 构建 scheduler，决定 pipeline 并行与 buffer 分配策略
                                                                    这里首先获取了 每个后端的缓冲区类型和后端的指针，方便接些来使用
                                                                        backend_buft 是一个 std::vector<ggml_backend_buffer_type_t> 类型的向量，用于存储每个后端的默认缓冲区类型
                                                                        backend_ptrs 是一个 std::vector<ggml_backend_t> 类型的向量，用于存储后端指针   
                                                                    创建两个图 gf_res_prev 和 gf_res_reserve 在后面实现图的复用或者其他高级功能会用到
                                                                          这里的 llm_graph_result 对象，用于存储图计算结果， max_nodes 表示图中的最大节点数，这个值决定了需要分配多少内存来存储图结构
                                                                          在 new llm_graph_result 对象的时候,有两个管家你的属性
                                                                                    ctx_compute.reset(ggml_init(params));   //计算图持有的张量容器ggml_contxt，
                                                                                    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false); //真正的计算图
                                                                                            在这个函数中，实例化了一个cgraph对象，可以看看cgrph的结构体定义：
                                                                                                                    struct ggml_cgraph {
                                                                                                                        int size;    // maximum number of nodes/leafs/grads/grad_accs
                                                                                                                        int n_nodes; // number of nodes currently in use
                                                                                                                        int n_leafs; // number of leafs currently in use
                                                                                                                        struct ggml_tensor ** nodes;     // 这里为啥要使用指针的指针呢？
                                                                                                                        struct ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
                                                                                                                        struct ggml_tensor ** grad_accs; // accumulators for node gradients
                                                                                                                        struct ggml_tensor ** leafs;     // tensors with constant data
                                                                                                                        int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot
                                                                                                                        struct ggml_hash_set visited_hash_set;
                                                                                                                        enum ggml_cgraph_eval_order order;
                                                                                                                    };
                                                                                                                    解释这里使用 指针的指针的必要性：
                                                                                                                        1.主要是为了这么使用 cgraph.nodes[i] ，从而这么设计的
                                                                                                                            struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false); // 创建图
                                                                                                                            gf->nodes[gf->n_nodes++] = tensor; // 添加节点
                                                                                                                            for (int i = 0; i < gf->n_nodes; i++) {  
                                                                                                                                struct ggml_tensor * node = gf->nodes[i];   // 访问节点
                                                                                                                                // 处理节点...
                                                                                                                            }    
                                                                                                                        2. 而且这里的 nodes 数组需要在运行时能够动态分配，这个问题在 python java 等支持动态数组的语言中没有感知，但是在c语言中这个功能要自己考虑怎么实现
                                                                                                                        3. 这里使用 指针的指针，这样 就可以在运行时再分配一个 指针数组， 把其地址挂到gf->nodes上，然后就可以gf->nodes[i] 访问 ggml_tensor 结构体了
                                                                                                                        4. 但是这个 指针数组 一经分配，大小也就确定了， 需要扩容就要整个重新分配一个更大的数组，然后把原来的数据拷贝过来
                                                                                                                        5. 从始至终这里的 cgraph 结构体是不会变的
                                                                                            cgraph初始化完成    
                                                                          到这里两个计算图 gf_res_prev gf_res_reserve 就初始化完成了，但是这个时候分配的空间里面存的都是 张量元数据 ggml_tensor ggml_tensor_object这些对象大小，但是这里面的 data 字段的指针还是空的，还没有分配真正的后端的内部
                                                                    
                                                                    然后就检查了是否启用 pipeline 并行 检查每个设备是否支持 async & events，如果不支持则关闭并行 pipeline 

                                                                    这里就开始创建调度器了：sched.reset(ggml_backend_sched_new(...))  在这个函数中， 创建了调度器用于管理多个后端和计算图的执行
                                                                    
                                                                        在上面的步骤中我们已经把原始的计算图cgraph结构体搞定了，但是我们并没有实例cgraph里面的张量，现在，这些都放在调度器对象中，
                                                                        在这个调度器中，我们要能快速的把原始图随意的切分做流水线，随意的备份做并行计算，这件事情怎么做呢? 这里的初始化有一个思想，
                                                                        
                                                                        我们在内存中搞一个大操场，在 scheduler中搞一个大操场，在这个操场上划分了两块区域，左边代表完全打散排布，右边代表组合成图排布
                                                                        
                                                                        在左边 我们假设 我们这个计算图可能会有 模型中张量的数量的8倍个节点，而且每个节点都被拆成子图，每个子图都有30个输入，每个输入跨设备要有个副本
                                                                        我们就按照这个方式在操场左边摆放好了所有张量的座位

                                                                        在右边我们假设一台机器上最多只有16张卡， 每张卡只会执行一张子图，那么最多有16张子图就好了
                                                                        我们依然给每个子图有30个输入张量，我们也按照这个方式在操场的右边摆好了座位

                                                                        现在我们 一些数组，左边的组数长度等于左边的座位数量，右边的右边的座位数量，这就相当于主席台上的笔记本，左边是名单，右边是组合结果
                                                                        后面需要调度，分割都是对着这个笔记本操作的

                                                                        至于座位什么要分这么多，没有为什么，就是先全占着，尽可能占上，反正座位占不了多少空间，后面操作 gf->nodes[i++] 就可以很方便的操作了

                                                                                这里有一个笔记本是记录张量副本的索引 hv_tensor_copies ， 它的使用方法很有机制：
                                                                                                1. hv_tensor_copies 是一个三维数组，定义为 [hash_set.size][n_backends][n_copies]
                                                                                                2. 这个数组存储了张量在不同后端和不同副本索引下的副本指针
                                                                                                3. 只有计算图中的节点或者叶子节点可能会有副本：
                                                                                                        3.1 当一个张量在一个后端上生成，但需要在另一个后端上使用时，会创建副本
                                                                                                        3.2 在计算图分割过程中，不同分割之间的边界张量需要副本
                                                                                                        3.3 当启用流水线并行时，为了支持不同阶段的并行执行，会创建多个副本
                                                                                                4. 只要一个张量有了副本，就会有数据一致性的问题需要解决，从这里也能够看出就是一个张量需要在多个设备上使用就会有多个副本
                                                                                                    但是不同副本的使用时机会有严格的依赖顺序，例如，把一个图分割两个子图，前半段在GPU上，后半段在CPU上，
                                                                                                    那么断裂出的张量都会在两个GPU CPU上各有一个副本
                                                                                                    现在GPU上算，完成后这些张量已经被赋值了，
                                                                                                    然后会把这些张量的data跨设备同步到CPU上，这里就是主要延迟点
                                                                                                    然后在CPU上继续算
                                                                                                5. 这就是张量副本的使用方法
                                                                                        
                                                                        scheduler 内部的操场布置好了之后， scheduler 中和真正的内存打交道的是 内存分配器，接下来会创建 内存分配器
                                                                            //创建图分配器， 内存分配器是一个 多层级的 分配系统，最上层是图分配器，负责整个计算图的内存分配   图分配器-张量分配器-线性块分配器
                                                                            sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);  
                                                                                                    在这里 会创建一个 图内存分配器的实例，
                                                                                                            图内存分配器里面 有几个数组， 长度和后端的数量一样， 分别记录 每个后端的buffer类型，每个后端的buffer地址
                                                                                                            最重要的还有个数组 记录 每个后端的能使用的 张量分配器
                                                                            到这里给把需要的 张量分配器创建好就算是 分配器初始还完成了
                                                            到这里就算是 调度器 初始化完成了



                                                                                                            


                                                                        基于这个思想，整个cgraph的初始化步骤大致为：
                                                                            函数参数和断言检查
                                                                            调度器结构体的分配和初始化
                                                                            哈希表的初始化
                                                                            各种数组和缓冲区的分配
                                                                            后端和缓冲区类型的初始化
                                                                            事件和内存分配器的创建
                                                                            调度器指针赋值


*/

/*  



*/

/*
// scheduler 就是 ggmll后端的调度功能的实现了
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


    