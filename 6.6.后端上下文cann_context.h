
/**
 * @brief Context for managing CANN backend operations.
 */
struct ggml_backend_cann_context {
    int32_t device;                  /**< Device ID. */
    std::string name;                /**< Name of the device. */
    std::string description;         /**< Description of the device. */
    aclrtEvent copy_event = nullptr; /**< Event for managing copy operations. */
#ifdef USE_ACL_GRAPH
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    std::unique_ptr<ggml_cann_graph> cann_graph;
    bool acl_graph_mode = true;
#endif
    cann_task_queue task_queue;
    bool async_mode;
    // Rope Cache
    ggml_cann_rope_cache rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = {nullptr}; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(int device)
        : device(device), name("CANN" + std::to_string(device)), task_queue(1024, device) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();

        async_mode = parse_bool(get_env("GGML_CANN_ASYNC_MODE").value_or(""));
        GGML_LOG_INFO("%s: device %d async operator submission is %s\n", __func__,
            device, async_mode ? "ON" : "OFF");
#ifdef USE_ACL_GRAPH
        acl_graph_mode = parse_bool(get_env("GGML_CANN_ACL_GRAPH").value_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n",
              __func__, device,
              acl_graph_mode ? "GRAPH" : "EAGER",
              acl_graph_mode ? "acl graph enabled" : "acl graph disabled");
#endif
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        task_queue.stop();
        if (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrtStream stream(int stream) {
        if (streams[stream] == nullptr) {
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        return streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrtStream stream() { return stream(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool>
        mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    static std::unique_ptr<ggml_cann_pool> new_pool_for_device(int device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool& pool() {
        if (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        return *mem_pool;
    }
};

#endif  // CANN_COMMON_H
