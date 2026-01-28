## Memory

Modern GPU architecture relies on a complex memory hierarchy that mirrors the structure of a CPU but at a much higher scale for parallel processing. This design is not a choice made for convenience. It is a fundamental necessity dictated by the laws of physics and the constraints of circuit design. To understand why your graphics card has different types of memory, we have to look at the physical trade-offs between speed, size, and cost.
### The Physical Trade-off: SRAM vs. DRAM
The foundation of the hierarchy lies in two different types of storage cells. Each serves a specific purpose based on its physical footprint and performance characteristics.

- **SRAM (Static RAM):** These cells are incredibly fast because of their complex control circuitry. However, that same complexity makes the cells physically large. Because they take up so much real estate on the silicon die, it is impossible to pack them in high densities without making the chip enormous and prohibitively expensive.
- **DRAM (Dynamic RAM):** These cells are designed for density. The cells are much smaller and simpler, which allows manufacturers to provide the massive capacities we see in modern VRAM. The trade-off is that DRAM is significantly slower than its SRAM counterpart.

### Why Hierarchies are Mandatory
This tension between capacity and latency is the exact reason why cache hierarchies exist. In an ideal world, every single compute unit on a GPU would sit directly next to a vast pool of ultra fast memory. Since physics makes this impossible, GPU designers have to compromise.

They place a very small amount of fast SRAM as close to the compute units as possible. This fast tier is then backed by progressively larger pools of slower memory located further away from the processing core. This distance is a critical factor because the further a signal has to travel, the longer the latency becomes.
### Maximizing System Throughput
The ultimate goal of this organization is to maximize overall system throughput. By keeping the most frequently used data in the tiny, fast pools near the logic gates, the GPU minimizes the time compute units spend waiting for information.

When the data is not found in the fast local cache, the system reaches out to the larger L2 cache or the massive external VRAM. This tiered approach ensures that even though the bulk of the memory is relatively slow, the processing cores remain fed with data as efficiently as possible. It creates a balance where the GPU can handle massive datasets without sacrificing the raw speed needed for real time computation.

Each level of the hierarchy serves a specific purpose in the data pipeline, starting from the furthest point and moving inward to the processing core.

- **Device Memory (VRAM):** In CUDA terminology, this is the off chip DRAM. In high performance cards, this is implemented as stacked High Bandwidth Memory or HBM. It hosts global memory and per thread local memory used for register spills. While it offers the largest capacity, it also has the highest latency.
- **L2 Cache:** This is a large SRAM cache that serves as a bridge between the device memory and the individual cores. It is physically partitioned into two parts. Each Streaming Multiprocessor (SM) connects directly to one partition and indirectly to the other through a high speed crossbar.
- **Distributed Shared Memory (DSMEM):** This layer represents the pooled shared memories of a physically close group of SMs, known as a Graphics Processing Cluster or GPC. It allows for efficient data sharing across a cluster of cores.
- **L1 Cache and Shared Memory (SMEM):** These two share the same physical SRAM storage on each SM. The L1 cache is a k-way set associative cache private to the SM, while Shared Memory is a programmer managed space. Software can often configure the relative split between these two based on the needs of the kernel.
- **Register File (RMEM):** This is the fastest storage in the entire system, located directly next to the compute units. Unlike CPUs, GPUs contain a massive number of registers. The total RMEM capacity is often equal in size to the combined L1 and Shared Memory storage.
### The Tensor Memory Accelerator (TMA)
Introduced with the Hopper architecture, the Tensor Memory Accelerator is a critical component for modern AI workloads. The TMA enables asynchronous data transfers between global memory and shared memory. It can also move data across shared memories within a cluster.

A key feature of the TMA is its support for swizzling. This is a technique used to reorganize data patterns to reduce bank conflicts, ensuring that memory access remains efficient during complex tensor operations.
### Key Takeaways for Optimization
Understanding this hierarchy leads to two fundamental rules for GPU programming and performance tuning:
1. **Proximity is Performance:** You must keep the most frequently accessed data in the highest levels of the hierarchy, specifically registers and shared memory.
2. **Avoid the Global Bottleneck:** Every trip to device memory (GMEM) is costly. Minimizing these accesses is the most effective way to improve the throughput of your kernels.


---

## GPU Compute: Inside the Streaming Multiprocessor

When we shift from storage to processing, the fundamental unit of execution is the **Streaming Multiprocessor (SM)**. On a high end chip like the Hopper H100, there are 132 of these units working in parallel. While the marketing might focus on "Graphics," in modern server class GPUs, these clusters serve purely as AI acceleration units.

### The Anatomy of the SM

Every SM is a self contained engine designed for massive throughput. It is physically divided into four quadrants, each housing a specific subset of compute units:

- **Tensor Cores:** These are specialized units built for matrix multiplications. They operate on small tiles of data at incredible speeds. Since large matrix multiplications are just collections of these smaller tile operations, leveraging Tensor Cores is the only way to reach peak AI performance.
- **CUDA Cores and SFUs:** The standard CUDA cores handle floating point operations like Fused Multiply Add (FMA). Alongside them, Special Function Units (SFUs) manage complex transcendental functions such as sine, cosine, and logarithms.
- **Warp Schedulers:** These are the brains of the SM. They issue instructions for groups of 32 threads, known as warps. Each scheduler can issue one warp instruction per cycle.
- **Load/Store (LD/ST) Units:** These circuits handle the movement of data between the compute units and the memory hierarchy, working in tandem with the Tensor Memory Accelerator.
### Hierarchy of Execution: SMs to GPCs
SMs are not scattered randomly across the die. They are organized into **Graphics Processing Clusters (GPC)**. On a full GH100 die, each GPC contains 18 SMs. However, you will notice that retail products often expose 132 or 114 SMs instead of the theoretical 144. This is because some SMs are fused off during manufacturing to improve yield. This physical reality is important for developers to remember when choosing cluster configurations for their kernels.

### Parallelism vs. Concurrency
One of the most important distinctions in GPU architecture is the difference between how many threads are running at once and how many are "in flight."

- **Parallelism (The 128 limit):** An SM can issue instructions from at most four warps simultaneously. This means exactly 128 threads are executing an instruction in true parallel at any given cycle.
- **Concurrency (The 2048 limit):** An SM can host up to 2048 concurrent threads. These threads are resident on the chip and ready to run. The hardware quickly schedules them in and out to hide latency. When one warp is waiting for data from memory, the scheduler swaps in another warp to keep the compute units busy.
### The "Speed of Light" and Power Throttling

The theoretical maximum throughput of a GPU is often called its **Speed of Light (SoL)**. This is the absolute upper bound of performance dictated by the physical characteristics of the chip. You can calculate this peak for specific data types, like bfloat16 or fp8, using a simple formula:

$$perf = freq\_clk\_max \times num\_tc \times flop\_per\_tc\_per\_clk$$

In plain English, this is the maximum clock frequency multiplied by the number of tensor cores and the operations they can perform per cycle.

However, this "speed of light" is not a fixed constant. In practice, the peak throughput depends on the actual clock frequency. If the GPU hits a thermal limit or a power ceiling, the hardware will throttle the clock speed. When the clock drops, your effective speed of light drops with it, meaning your kernel will run slower regardless of how well the code is optimized.