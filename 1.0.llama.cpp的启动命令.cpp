/*
    我们主要目标是  从启动命令中 梳理出 整个框架的 启动加载和执行的过程，而不关注具体的 函数实现和细节。


    1.执行命令 ./build/bin/llama-cli -m /home/chenchen/models/gemma-3-4b-it-f16.gguf -p '用一句话介绍你自己。' -no-cnv -ngl 100 -sm none -mg 0 
    2.就会从这里的 main 函数开始执行
    3.   依次执行了   common_init();   // 只是初始化了日志工具，时钟对齐
                    llama_backend_init();   // 只是返回了一个 context 指针
                    llama_numa_init(params.numa);  // 初始化NUMA，这里的调用，触发了 后端初始化
                        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); 
                            ggml_backend_dev_count()
                                get_reg()
                                    static ggml_backend_registry reg;   //会执行 ggml_backend_registry 的构造方法
                                        这里 ggml_backend_registry reg 有了两个成员变量，
                                            std::vector<ggml_backend_reg_entry> backends;  这个装所有具体后端描述信息
                                            std::vector<ggml_backend_dev_t> devices;       这个装所有的具体卡的描述信息
                                        这里 执行 ggml_backend_registry 构造方法的目的就是 获取到这两个信息
                                        register_backend(ggml_backend_cann_reg());   // 触发 cann后端的注册方法
                                            ggml_backend_cann_reg() 
                                                ggml_cann_info()
                                                    ggml_cann_init(){  //调用到了初始化cann后端
                                                       在这个方法 使用 aclrtGetDeviceCount 数了当前有几张卡， aclrtMemGetAllocationGranularity 每张卡的内存信息
                                                       把这些信息 写入 ggml_cann_device_info 返回
                                                    }
                                                这里创建了一个 ggml_backend_reg 对象 cann_reg,  cann_reg 持有一个 ggml_backend_cann_reg_context 对象 reg_ctx 
                                                每个 reg_ctx 对象 只有一个向量字段 std::vector<ggml_backend_dev_t> devices;
                                                每个 ggml_backend_dev_t 对象 持有自己这张卡的 dev_ctx,
                                                每个 dev_ctx 用三个字段描述这张卡     int device; std::string name; std::string description;
                                                返回了 cann_reg    到目前为止 还是只是检测了卡信息，没有初始化
                                            这里拿到了 cann_reg 后，直接 backends.push_back(cann_reg);
                                            然后 遍历了 cann_reg， 逐个调用 register_device(ggml_backend_reg_dev_get(reg, i))
                                                                                             ggml_backend_reg_dev_get(reg, i)
                                                                                                reg->iface.get_device(reg, index)  //这里绑的是cann的实现函数
                                                                                                    ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t index)
                                                                                                        在这里取出了 devices[i] 的 dev_ctx
                                                                  devices.push_back(cann_reg);
                                        到这里 ggml_backend_registry reg 的两个成员变量 都有了值
                        就可以继续 执行 ggml_backend_cpu_numa_init() 完成 cpu后端的 numa 优化了，  
                        到此 cann 后端的8张卡的reg 信息已经放到全局的 reg注册表了，get_reg() 就可以i访问，但是还是没有初始化

                    实际上默认无需优化 numa，直接跳到这里了，上面的一堆都没执行
                    
                    common_init_from_params(params); 这里开始真正的初始化
                                model = llama_model_load_from_file(params.model.path.c_str(), mparams);
                                cparams = common_context_params_to_llama(params);
                                lctx = llama_init_from_model(model, cparams);
                                            ctx = new llama_context(*model, params);   // 这里开始初始化上下文，在llama_context的构造方法中完成了大量的初始化工作
                                                            1.初始化了 GPU后端，加速卡后端，CPU后端，线程池等
                                                                    for (auto * dev : model.devices) {    //在npu上，到这里进来了，检测到了有 8 张卡
                                                                        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);    //从这里开始初始化  在这里循环了8次
                                                                        backends.emplace_back(backend);
                                                                    }

                                                                    关键在 ggml_backend_dev_init() 里面
                                                                                device->iface.init_backend(device, params); 
                                                                                就是调用了 ggml_backend_cann_device_init()
                                                                                                ggml_backend_cann_init(ctx->device);  // 这里初始化了cann后端实例
                                                                                                        aclInit(nullptr);
                                                                                                        cann_ctx = new ggml_backend_cann_context(device);
                                                                                                        cann_backend = new ggml_backend{   // 终于初始化了 cann 后端
                                                                                                                                                        ggml_backend_cann_guid(),
                                                                                                                                                        ggml_backend_cann_interface,
                                                                                                                                                        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                                                                                                                                                        cann_ctx  //最关键的实例
                                                                                                                                                    };
                                                                                                        这里我们看看 cann_ctx 中究竟装了些什么，  cann_ctx 中有这张卡的详细信息 和 task_queue, stream_pool, cache 等
                                                                                                        但是这里初始化的时候，只是给几个简单的字段赋了值：
                                                                                                                device(device)
                                                                                                                name("CANN" + std::to_string(device)), 
                                                                                                                task_queue(1024, device)
                                                                                                                description = aclrtGetSocName()
                                                                                                                async_mode = parse_bool(get_env("GGML_CANN_ASYNC_MODE").value_or(""))
                                                                                                                acl_graph_mode = parse_bool(get_env("GGML_CANN_ACL_GRAPH").value_or("on"))
                                                                                                        其他的运行时资源还没有初始化
                                                                    返回一个 ggml_backend_t 对象 backend 然后这个对象会被 放入 llama_context 的一个持有数组对象中  backends.emplace_back(backend)
                                                            
                                                            2.然后，类似的调用流程 初始化了  CPU backend
                                                                backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);  // 在这里完成了 cpu 后端的初始化
                                                                backends.emplace_back(backend_cpu);

                                                            3.到此为止，有了 9 个后端，然后把这 9 个后端的 名为ggml_backend_set_n_threads的函数接口到 放到一个里表中，方便后面给每个后端设置线程数

                                                            4.设置 abort 回调  这里埋入了一个中断函数指针，该函数指针在推理时每个node结束都会扫一眼，如果有中断信号，就该图退出推理。

                                                            8. 分配输出缓冲区（logits/embd）    调用  output_reserve  分配 logits/embedding 缓冲区   这是构造过程中第一处大内存分配  输出 buffer 的大小 ≈  n_seq_max * (n_vocab + n_embd) * sizeof(float) 
                                                            9. 初始化 Memory 模块（KV cache）    根据参数分配 KV cache（存 K/V 张量用来复用注意力）  会预分配 KV tensor 的 backend buffer
                                                                                                这类的详情见 llama.cpp的kv-cache.cpp
                                                            10. 构建 scheduler，决定 pipeline 并行与 buffer 分配策略
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

                                                                        这里有一个笔记本是记录张量副本的索引hv_tensor_copies， 它的使用方法很有机制：
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
                                                                                        
                                                                        todo: 接下来就是要分配真正后端的内存了，看看还会这么浪费的分配占用吗？

                                                                        基于这个思想，整个cgraph的初始化步骤大致为：
                                                                            函数参数和断言检查
                                                                            调度器结构体的分配和初始化
                                                                            哈希表的初始化
                                                                            各种数组和缓冲区的分配
                                                                            后端和缓冲区类型的初始化
                                                                            事件和内存分配器的创建
                                                                            调度器指针赋值






                                                            11. 检测/开启 Flash Attention   这是兼容性检测 因为不是所有的设备都支持flashattention
                                                                                                        如果  params.flash_attn_type == AUTO ：
                                                                                                        临时构造一个 test graph
                                                                                                        遍历所有节点，检查 FA 算子 (Flash Attention) 的设备分配是否与 KV 层一致
                                                                                                        如果 mismatch，禁用 FA
                                                                                                        否则开启
                                                            12. 预留 graph (prompt + token gen)   提前预分配 graph 内部用 buffer（算子中间结果） 避免推理时  ggml alloc  的频繁 re malloc  此处会做第二波大内存分配。 
                                                            13. 打印 compute buffer 内存占用      打印每个设备占用多少显存/内存
                                                            14. 构建 graph result 容器          尽量复用结果容器，用于保存 graph 的执行结果和 output tensor。 避免反复 new/delete。
                                                            15. 构造完成    至此， llama_context  进入可用状态                                                                                    
                common_init_from_params的初始化就完成了
    



                    */


int main(int argc, char ** argv) {

    common_init();   // 通用工具初始化
    llama_backend_init();   // 初始化后端
    llama_numa_init(params.numa);  // 初始化NUMA
    llama_init = common_init_from_params(params); // 真正的大量初始化后端、模型和上下文

    //这些都是获取指针的操作
    model = llama_init.model.get();  // 获取模型
    ctx = llama_init.context.get();  // 获取上下文
    mem = llama_get_memory(ctx);   // 获取内存
    vocab = llama_model_get_vocab(model);  // 获取词汇表
    chat_templates = common_chat_templates_init(model, params.chat_template);  // 初始化聊天模板
    ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);   // 获取CPU设备
    ggml_backend_dev_backend_reg(cpu_dev);   // 获取CPU后端注册
    ggml_threadpool_new_fn(&tpp);  // 创建线程池
    llama_attach_threadpool(ctx, threadpool, threadpool_batch);   // 关联线程池

    
     // 处理提示词
     // 格式化系统提示词
     // 格式化用户提示词
     // 初始化组注意力相关参数

    // 主循环
    while ((params.n_predict != 0 && !false) || params.interactive) {
        // 处理上下文
        // 上下文移位
        // 自扩展上下文
        // 模型推理
        llama_decode(ctx, llama_batch_get_one(&embd[0], embd.size()))
        // 生成新token
    }


    // 释放资源
    common_sampler_free(smpl);
    llama_backend_free();
    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}