// 管理“所有后端”的是内部的全局 ggml_backend_registry（在 ggml-backend-reg.cpp 里 static ggml_backend_registry reg;）
/*
这个文件 就是 实现 一个 ggml_backend_registry
这是一个全局单例
它里面 就两个字段
    一个存储所有的后端描述符 的 vector
    一个存储所有的设备 的 vector
*/

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;  //存储所有的后端 的 vector
    std::vector<ggml_backend_dev_t> devices;        //存储所有的设备 的 vector
                            // 在构造方法的就 把这两个字段填充好了
    // 构造函数
    ggml_backend_registry() {
        register_backend(ggml_backend_cann_reg());
        register_backend(ggml_backend_cpu_reg());
        //...  这里还有其他的 后端，但是 在编译的时候 通过配置变量 就能让 只有这两个可见
        //  这里 执行 ggml_backend_cann_reg() 返回的是 cann的设备类型描述对象 reg
    }

    // 后端注册核心方法
    void register_backend(ggml_backend_reg_t reg) {
        backends.push_back(reg);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            // 这里会 通过 cann的设备类型描述对象 reg 扫描出有几张卡
            ggml_backend_dev_t device = ggml_backend_reg_dev(reg, i);   // 这里返回的就是 每张卡的 设备对象
            devices.push_back(device);
        }
    }

    // 卸载后端方法
    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        // 查找并移除后端
        auto it = std::find_if(reg);

        // 移除关联的设备
        devices.erase(reg);
        backends.erase(it);
    }
};

// 全局单例注册表访问器
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// 一些便捷API 实现
ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) 

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) 

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) 

ggml_backend_t ggml_backend_init_best(void) 

void ggml_backend_load_all() 