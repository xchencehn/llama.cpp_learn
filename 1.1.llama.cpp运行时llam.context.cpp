/*

llama_context  是 llama.cpp 运行时的核心。
它包括  输出管理、图管理、状态存取、训练、性能统计 等功能，它是 llama.cpp的推理运行环境 Runtime Context

llama_context 的初始化
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
            创建两个图 gf_res_prev 和 gf_res_reserve 在后面实现图的复用或者其他高级功能会用到
                    到这里两个计算图 gf_res_prev gf_res_reserve 就初始化完成了，但是这个时候分配的空间里面存的都是 张量元数据 ggml_tensor ggml_tensor_object这些对象大小，但是这里面的 data 字段的指针还是空的，还没有分配真正的后端的内部
            然后就检查了是否启用 pipeline 并行 检查每个设备是否支持 async & events，如果不支持则关闭并行 pipeline 
            这里就开始创建调度器了：sched.reset(ggml_backend_sched_new(...))  在这个函数中， 创建了调度器用于管理多个后端和计算图的执行
            
            上面的这段内容详情见 2.3.ggml的调度器实现ggml-backend.cpp


    11. 检测并决定是否启用 Flash Attention  检测方法如下J：
                这里 flash_attn_type == AUTO 说明用户没明确开/关 FA，系统要自己判定能不能用
                graph_reserve(...) 会预构建一个小计算图 (gf)，主要用于检测 Flash Attention 节点会被安排在哪个 device 上
                遍历图中所有节点，找到 Flash Attention 节点 (算子 GGML_OP_FLASH_ATTN_EXT)
                查找这个 FA 节点被分配到的后端 (device_fa)。
                根据节点名字解析出它对应的模型层号il  然后根据 模型层号il 模型层权重被放在哪个 device 上 
                看看 如果 Flash Attention 节点和层权重分在不同 device，说明算子不被支持（或者落到了别的后端）, 就让 标记 mismatch 赋值为 true 
                只要检测到 标记 mismatch 赋值 为true   Flash Attention 直接关闭 

    12. 构图测试，这里分别测试 后端的内存构建两张图会不会出问题，一张是 prompt processing 阶段的 pp图，一张是 token generation 阶段的 tg图
                首先测试为 prompt processing 阶段准备一个 pp图 能否成功
                    构图的函数是被封装在 llama_context 中的 graph_reserve
                        graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
                            gf_res_prev->reset();   // 这里会重新构图
                            res_prev->reset();      // 这两个都是布局操场
                            auto * gf = model.build_graph(gparams);   // 这个是真正的模型的计算图
                            ggml_backend_sched_reserve(sched.get(), gf)  // 这里是真正的给 gf 分配buffer内存
                                ggml_gallocr_reserve_n   // 这里使用 galloc 分配内存
                                    ggml_backend_buft_alloc_buffer  // 这就是后端的分配内存的函数
                                        这个函数的作用就是给计算图分配真实的buffer
                                        这里传入的图是一个完整的计算图，但是实际上他可能是已经被规划好了，切分成了多块在多个后端上计算
                                        这里 是 galloc 来负责分配buffer, 在 scheduler 初始化的时候，我们就初始化好了 galloc
                                        galloc 里面有几个数组，记录了每个后端是怎么类型 持有后端指针，buffer指针
                                        我们遍历图的每个node的时候，自然会知道他该到那个后端执行
                                        这样我们就可以给每个node 和 leaf 的张量的data字段指向真正的buffer中的地址
                                        但是在指定之前大段的代码都是在计算每个后端上分配到这里的那块buffer够不够大
                                        够大就直接使用，不够大就整块free, 然后重新 alloc
                                    
                                        大致流程就是： 清空 → 模拟分配 offset → 根据统计的 max_size 真正分配 buffer（或者复用旧 buffer）
                    pp图构图，内存分配测试就完成了
                然后测试为 token generation 阶段准备一个 tg图 测试内存能否分配成功
                    和上面的一样的过程， 这样也只是测试构图的时候给图分配足够的真实内存，会不会出问题，但没有执行图
                TG图测试没问题，就重新再构图成为PP图，当然这次不是为了测试，而是马上就要执行这个pp图了
        构图 内存 分配测试完成

    13. 计算 buffer 大小并打印信息
            遍历所有 backend，打印每个设备计算 buffer 的大小（MiB）
            打印 graph 概况
            如果 PP 和 TG 的节点数相同，就直接打印；否则打出详细对比
            打印切分数，切分数表示调度器为了让 buffer 合适，将计算图拆分成多少段执行

    14. 构造完成    至此， llama_context  进入可用状态   
*/



/*

llama_context 会每次通过执行 llama_decode() 函数来执行模型的推理

在执行 encode 或者 decode 过程时，都会按照如下步骤：
        1. 首先 output_reserver() 看看 输出缓冲区够不够，不够的话可以扩容
        2. 调用 process_ubatch() 
                在这个函数中，是把一个微批次（ubatch）的输入数据传进来，构建或者复用一张计算图 graph，调用 graph_compute 把 graph 交给调度器跑完，然后把结果保存到 llm_graph_result 里
                1. 调用 memory 模块，可能调整 KV 位置
                2. 取上一次的 graph result，在上一次执行的cgraph对象中保留了上一次的结果
                3. 生成 Graph 参数，构造一个llm_graph_params，里面包含：   这些参数决定 graph 的拓扑结构
                                模型结构、超参数、运行时参数
                                batch 输入（ubatch）
                                memory context
                                graph 类型
                                输出数量
                                回调函数 
                4. 判断是否能复用之前的 graph, 如果参数完全一样  就可以直接复用上次的 graph  否则调用 model.build_graph()重新生成一张新的计算图  
                   第一次prompt需要构建新图，第一次decode需要构建新图，后面就可以复用了

                5. 设置输入张量数据 res >set_inputs(&ubatch) 这里就是把 数据复制到 backend buffer 中了
                        这里的 ubatch 是怎么选出来的？ 这里有两种解释 1. 同一个序列 的 prompt 一次要处理 256 长的一段，这就是一个 ubatch
                                                                    2. 256个序列，每人生成一个 next_token ， 这就是一个 ubatch
                
                6. 调用 graph_compute
                                这里会调用 ggml_backend_sched_graph_compute_async 让算子在 CPU/GPU/NPU backend 上执行 
                                1. 确定线程数 这里每个序列的推理使用一个线程，如果当前一次要对 3 个token 向前推理一步，那么就使用3个线程
                                2. 异步使用后端执行计算 ggml_backend_sched_graph_compute_async(sched.get(), gf);
                                            在这里面，就是让 调度器 sched 去执行 计算图 gf 的计算
                                            调度器根据 graph 的节点和张量分配，把算子调度到相应设备 backend 去执行                                                         
                                            这一步就是  真正把注意力、矩阵乘法、激活函数等  GPU/CPU 内核调起来。
                                            这一步具体的细节在 ggml_backend.cpp 文件中有详细的介绍 
                7. 返回结果
                    return res; 在这个llm_graph_result  res中可以取出 logits embeding 等信息res >get_logits() 、res >get_embd() 、res >get_embd_pooled()



5. 计算完输出的时候，有可能输出的token顺序 和 输入的 token 顺序不一致，所以还会调用一次 output_swaps 如果不一致就会 重新交换 logits/embd 内部的行。
保证对外暴露的  get_logits_ith(i) 、 get_embeddings_ith(i)  一定是正确对应。




*/

