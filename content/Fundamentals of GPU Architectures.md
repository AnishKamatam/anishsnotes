At the highest level, a GPU performs two essential tasks:
1. Move and store data (the memory system)
2. Do useful work with the data (the compute pipelines)

The block diagram of H100 below reflects this division: components in blue represent memory or data movement, while components in red are compute (hot) units.

![[Pasted image 20260128131608.png]]
Figure 1: Model of the NVIDIA Hopper H100 GPU

# Memory

GPU memory isn't a single flat space. It is a hierarchy dictated by circuit design. You generally have two choices when building memory:
- **SRAM (Static RAM):** This is the high performance option. It uses complex circuitry to achieve incredible speeds. However, that complexity makes the cells physically large. You can only fit a tiny amount of it on the chip before you run out of space.
- **DRAM (Dynamic RAM):** This is the high capacity option. It is dense and simple, which is why we can have 80GB of VRAM on a single board. The downside is that it is significantly slower than SRAM.
Since we cannot have a vast pool of ultra fast memory, designers use a hierarchy. They place a small amount of fast SRAM close to the compute units and back it with larger, slower pools of DRAM further away. This maximizes **throughput** by ensuring the math engines stay fed with data.

In the H100 architecture, data travels through five main levels. As you move closer to the compute units, bandwidth increases by orders of magnitude while capacity shrinks.
### 1. Device Memory (VRAM)
This is the off-chip HBM (High Bandwidth Memory). It serves as the "Global Memory" for the GPU. It is physically separate from the GPU die but packaged on the same board. It is the largest pool but also the slowest.
### 2. L2 Cache
This is a large SRAM cache built directly on the die. It is partitioned into two halves. Each Streaming Multiprocessor (SM) connects directly to one side and indirectly to the other through a crossbar.
### 3. Distributed Shared Memory (DSMEM)
This is a newer layer where groups of physically close SMs can pool their shared memory. It allows for efficient data exchange within a cluster of cores.
### 4. L1 Cache and Shared Memory (SMEM)
Each SM has its own private pool of fast SRAM. The hardware allows software to configure how much of this space is used as a traditional L1 cache versus programmer-managed "Shared Memory."
### 5. Register File (RMEM)
The fastest storage in the system. Registers sit right next to the compute units and are private to individual threads. On GPUs, the register file is massive, often equal in size to the L1/SMEM combined.

**The Golden Rule:** Keep your most frequently used data in the registers or SMEM. Every trip back to Global Memory is a performance killer.