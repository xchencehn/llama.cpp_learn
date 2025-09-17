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
                                                                                            在初始化张量容器的时候，计算并分配了图计算所需的元数据内存大小  总内存大小 = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false)
                                                                                                                                                     = 所有张量结构体的开销总和 + 图结构本身的开销
                                                                                                                                                     = (GGML对象的基本大小 + 张量结构体的大小) * max_nodes + (GGML对象的基本大小 + 对齐后的图结构字节大小)
                                                                                                                                                                                                                                 图结构字节大小 = ggml_graph_nbytes() 函数计算图结构所需的字节大小。
                                                                                                                                                                                                                                 它模拟了内存分配过程，通过 incr_ptr_aligned 函数累加各部分内存需求：
                                                                                                                                                                                                                                    1.sizeof(struct ggml_cgraph) ：图结构体本身的大小
                                                                                                                                                                                                                                    2.size * sizeof(struct ggml_tensor *) ：节点指针数组的大小
                                                                                                                                                                                                                                    3.size * sizeof(struct ggml_tensor *) ：叶子节点指针数组的大小
                                                                                                                                                                                                                                    4.hash_size * sizeof(int32_t) ：使用计数数组的大小
                                                                                                                                                                                                                                    5.hash_size * sizeof(struct ggml_tensor *) ：哈希键数组的大小
                                                                                                                                                                                                                                    6.如果需要梯度( grads=true )，还包括梯度和梯度累加器数组的大小
                                                                                                                                                                                                                                    7.ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t) ：位集的大小
                                                                                                                                                                                                                                    其中 hash_size = ggml_hash_size(size * 2) 是哈希表的大小，通常是图节点数的两倍。
                                                                                                                                                                                                                                在这里 每一次计算的指针都是 incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // nodes 移动的
                                                                                                                                                                                                                                incr_ptr_aligned中 每一段都会对内存对齐，所谓对齐就是将指针移动到下一个sizeof(struct ggml_tensor *) 的倍数的位置上，这样可以提高内存访问的效率，避免访问未对齐的内存导致的性能下降
                                                                                            在初始计算图对象的时候，
                                                                                                        首先调用 ggml_graph_nbytes 函数计算整个计算图结构所需的内存大小
                                                                                                        在刚才创建的 ggml_contxt 上下文对象中创建一个类型为 GGML_OBJECT_TYPE_GRAPH 的对象
                                                                                                        然后初始化了计算图的内存布局：
                                                                                                                +-------------------+
                                                                                                                | ggml_cgraph       |  计算图结构体
                                                                                                                +-------------------+
                                                                                                                | nodes_ptr         |  节点指针数组
                                                                                                                +-------------------+
                                                                                                                | leafs_ptr         |  叶子节点指针数组
                                                                                                                +-------------------+
                                                                                                                | use_counts_ptr    |  使用计数数组
                                                                                                                +-------------------+
                                                                                                                | hash_keys_ptr     |  哈希键数组      这里使用hash表是为了快速的索引到所有的节点和叶子节点
                                                                                                                +-------------------+
                                                                                                                | grads_ptr         |  梯度指针数组（可选）
                                                                                                                +-------------------+
                                                                                                                | grad_accs_ptr     |  梯度累加器数组（可选）
                                                                                                                +-------------------+
                                                                                                                | hash_used         |  哈希使用位集合
                                                                                                                +-------------------+
                                                                                            cgraph初始化完成    
                                                                          到这里两个计算图 gf_res_prev gf_res_reserve 就初始化完成了，但是这个时候分配的空间里面存的都是 张量元数据 ggml_tensor ggml_tensor_object这些对象大小，但是这里面的 data 字段的指针还是空的，还没有分配真正的后端的内部
                                                                    然后就检查了是否启用 pipeline 并行 检查每个设备是否支持 async & events，如果不支持则关闭并行 pipeline 
                                                                    这里就开始创建调度器了：sched.reset(ggml_backend_sched_new(...))
                                                                         详细的解剖这里的 创建过程：




                                                                                                                                                                                                                                                                                                                 
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