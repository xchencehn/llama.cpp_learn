/*

llama_context  是 llama.cpp 运行时的核心。
它包括  输出管理、图管理、状态存取、训练、性能统计 等功能，它是 llama.cpp的推理运行环境 Runtime Context


1. llama_context的设计目标:
        1.通过它能获取到模型的 权重
        2.通过它能获取到 CPU  GPU backend 调度器等 核心资源
        3.通过它能获取到模型执行过程中的 kv cache 输出的logits embeding 等信息
        4.通过它能构建计算图，调度图的执行，还能随时中断图的执行
        5.在框架层能对当前性能情况进行监控这控制

2. 为了完成上面的需求，我们需要设计一些它的成员变量来存储一些关键的信息
        1.模型与参数  model 绑定的静态模型  cparams 运行时上下文设置（比如 batch size、seq 上限、线程数、attention 类型）  llama_adapter_loras 额外的LoRA/Adapter插件
        2.内存管理   memory KV cache管理模块   buf_output 推理输出用的缓冲区(保存logits+embedding)   
        3.计算调度   backends 所有设备后端GPU/NPU/CPU  backend_cpu CPU后端  sched 调度器负责把计算图分配到各个backend上执行 gf_res_reserve 保存上一次或预留的计算结果 threadpool 线程池
        4.输出组织   output_ids batch中每步输出tokend的id output_swaps 用于推理后输出顺序的重排
        5.性能与状态   一堆 t_eval_us 计时器 n_eval 计数器  保存/恢复相关 API

3. 这么多的功能，整个框架的启动基本都规划在了 这个类的构造函数中
        llama_context  的构造函数是 llama.cpp  里最复杂的部分之一，它要一次性准备好推理所需的几乎全部运行时环境，因此步骤复杂且依赖关系很多。
        1. 日志/时间初始化  
        2. 合并参数 →  cparams      把  llama_context_params  和模型自身的  hparams  合并 如果用户传了参数 → 用用户的 否则用模型训练时的默认值 有些情况带逻辑修正（例如  n_seq_max = 0 → 至少要1 ） 
        3. 设置注意力/rope/flash/pooling 等模式  
                                  设置 attention 类型（causal / 非 causal） 设置 rope scaling（YARN / NONE） 设置 flash attention 是否启用 设置 pooling 类型   设置 batch 大小限制  
                                  causal attention 的 batch 限制在  n_ctx 
                                  batch < GPU 内核 pad 大小则提升到合适值
        4. 检查环境变量（图复用）     检查是否禁用 graph 重用，调试时可能需要强制重建计算图。
        5. 打印参数日志             n_seq_max, n_ctx, n_batch, n_ubatch, causal_attn, flash_attn, kv_unified…
        6. 初始化 backends（GPU → ACCEL → CPU）并绑定控制函数  
                                  初始化 GPU backend      这里具体的详见 1.0.llama.cpp的启动命令 在底层调用了 CUDA/CANN/自定义 runtime API 初始化
                                  初始化 ACCEL backend     这里跳过
                                  初始化 CPU backend      CPU backend 一定要有，是「保底后端」。
                                  给每个 backend 绑定一个「调线程数」的控制函数。 防止运行时线程乱切。


        7. 设置 abort 回调    注册中断检查函数，在每个 node 结束时调用，支持「人为终止推理」。
        8. 分配输出缓冲区（logits/embd）    调用  output_reserve  分配 logits/embedding 缓冲区   这是构造过程中第一处大内存分配  输出 buffer 的大小 ≈  n_seq_max * (n_vocab + n_embd) * sizeof(float) 
        9. 初始化 Memory 模块（KV cache）    根据参数分配 KV cache（存 K/V 张量用来复用注意力）  会预分配 KV tensor 的 backend buffer
                                            这类的详情见 llama.cpp的kv-cache.cpp
        10. 构建 scheduler，决定 pipeline 并行与 buffer 分配策略  
                                                按 backend 数量构造调度器 会决定：
                                                        buffer 如何分配
                                                        是否启用 pipeline 并行
                                                        检查每个设备是否支持 async & events，如果不支持则关闭并行 pipeline 
        11. 检测/开启 Flash Attention   这是兼容性检测 因为不是所有的设备都支持flashattention
                                                    如果  params.flash_attn_type == AUTO ：
                                                    临时构造一个 test graph
                                                    遍历所有节点，检查 FA 算子 (Flash Attention) 的设备分配是否与 KV 层一致
                                                    如果 mismatch，禁用 FA
                                                    否则开启
        12. 预留 graph (prompt + token gen)   提前预分配 graph 内部用 buffer（算子中间结果） 避免推理时  ggml alloc  的频繁 re malloc  此处会做第二波大内存分配。 
        13. 打印 compute buffer 内存占用      打印每个设备占用多少显存/内存
        14. 构建 graph result 容器          尽量复用结果容器，用于保存 graph 的执行结果和 output tensor。 避免反复 new/delete。
        15. 构造完成    至此， llama_context  进入可用状态：


4. 初始化完成后，在执行过程中，也是围绕这个这个 llama.context 对象来进行完成推理
        1. 在执行 encode 或者 decode 过程时，都会按照如下步骤：
                1. 首先 output_reserver() 看看 输出缓冲区够不够，不够的话可以扩容
                2. 调用 process_ubatch() 
                            在这个函数中，是把一个微批次（ubatch）的输入数据传进来，构建或者复用一张计算图 graph，调用 graph_compute把graph交给调度器跑完，然后把结果保存到 llm_graph_result 里
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
                                            还要调用调度器alloc_graph为 graph 在 backend 上分配 buffer 重新分配内存 👉这是优化的关键

                            5. 设置输入张量数据 res >set_inputs(&ubatch) 这里就是把 数据复制到 backend buffer 中了
                            
                            6. 调用 graph_compute
                                            这里会调用 ggml_backend_sched_graph_compute_async 让算子在 CPU/GPU/NPU backend 上执行 
                                            1. 确定线程数 这里每个序列的推理使用一个线程，如果当前一次要对 3 个token 向前推理一步，那么就使用3个线程
                                            2. 异步使用后端执行计算 ggml_backend_sched_graph_compute_async(sched.get(), gf);
                                                        在这里面，就是让 调度器 sched 去执行 计算图 gf 的计算
                                                        调度器根据 graph 的节点和张量分配，把算子调度到相应设备 backend 去执行                                                         
                                                        这一步就是  真正把注意力、矩阵乘法、激活函数等  GPU/CPU 内核调起来。
                                                        这一步具体的细节在 ggml_backend.cpp 文件中有详细的介绍， 这里是 llama_context的内容
                            7. 返回结果
                                return res; 在这个llm_graph_result  res中可以取出 logits embeding 等信息res >get_logits() 、res >get_embd() 、res >get_embd_pooled()



5. 计算完输出的时候，有可能输出的token顺序 和 输入的 token 顺序不一致，所以还会调用一次 output_swaps 如果不一致就会 重新交换 logits/embd 内部的行。
    保证对外暴露的  get_logits_ith(i) 、 get_embeddings_ith(i)  一定是正确对应。

    */