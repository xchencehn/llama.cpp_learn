/*
我们主要目标是  从启动命令中 梳理出 整个框架的 启动加载和执行的过程，而不关注具体的 函数实现和细节。


1.执行命令 ./build/bin/llama-cli -m /home/chenchen/models/gemma-3-4b-it-f16.gguf -p '用一句话介绍你自己。' -no-cnv -ngl 100 -sm none -mg 0 
2.就会从这里的 main 函数开始执行
3.依次执行了
    common_init();   // 只是初始化了日志工具，时钟对齐
    llama_backend_init();   // 只是返回了一个 context 指针
    llama_numa_init(params.numa);  // 默认无需优化 numa，直接跳到这里了，上面的一堆都没执行
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
        又做了一些 参数上的 是否开启 窗口上下文，lora，采样参数 等 配置

        执行了一次预热计算， 先跑一次空的推理，让权重加载到显存/内存、初始化一些 kernel，最核心的就是他执行了如下函数：
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()))
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)))
            这里会根据模型的架构来执行，我们只看 llama_decode 
                **** 关于预热
                        在初始化阶段预热，就是喂给 llama_decode 一小批无关紧要的 token（比如 BOS/EOS，甚至 0），不是真的要生成输出，而是“跑一遍计算图”，
                        这样可以完成如下工作：

                            1. 把模型权重和算子 load 到 GPU 内存中，触发 lazy allocation
                            2. 让编译优化器 (ggml / CUDA 内核 ) 做 JIT 编译和内核缓存  第一次调用 kernel 会编译 PTX → GPU SASS
                            3. 可以先填充一次 cache：避免首次真正 decode 时遇到冷启动延迟
                            4. 最后又 llama_memory_clear() 把 KV 缓存清空，恢复到「没跑过」的干净状态 不影响后面的真实推理
                        有 warmup 后，耗时只发生在加载阶段
        预热完成        
    common_init_from_params的初始化就完成了


    然后做了一些获取指针的操作拿到了： model ctx mem vocab chat_templates 对象

    就开始了推理的主循环, 这段循环伪代码大致如下：

    while (还没停) {
        // 1. 如果 embd 有内容 → 送入 llama_decode
        if (!embd.empty()) {                                    // 注意这里 embd 是一个暂存区，只存这一次要送进 llama_decode 的 token 批次
            for ( batch in embd ) {                             // 如果prompt太长，一次可能处理不完，所以这里也分batch处理
                llama_decode(ctx, batch_tokens);                // llama_decode 的详情在 llama_context.cpp 中
                n_consumed += batch_size                        // 记录处理了多少个prompt token 了
            }
            embd.clear(); // 清空等待区
        }

        // 2. 采样下一个 token 放入 embd
        if (n_consumed >= embd_inp.size()) {                   // 确定 prompt 用完了，开始采样
            new_id = common_sampler_sample(smpl, ctx, -1);     // 采样
            embd.push_back(new_id);                            // 由于前面 embd 已经清空了，所以现在里面就只有一个 new_id
        } 


        // 3. 检查停止条件 (EOG)
        if (llama_vocab_is_eog(vocab, 最后一个token)) {
            print(" [end of text]");
            break; // 完整结束
        }

    
    } // end while

    推理完成释放资源

*/


int main(int argc, char  argv) {

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