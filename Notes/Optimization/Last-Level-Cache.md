### Introduction

- The trick is not to put the cache near the processor. It’s counterintuitive, but it works. Most high-end embedded processors, like an Arm Cortex A-series, will have L1 and L2 caches for each CPU core. Sometimes, the processor complex has an L3 cache as well that’s shared among all the CPU cores. That all works fine. No adjustment is necessary. 

- Now, add a fourth cache – a last-level cache – on the global system bus, near the peripherals and the DRAM controller, instead of as part of the CPU complex. The last-level cache acts as a buffer between the high-speed Arm core(s) and the large but relatively slow main memory. 

- This configuration works because the DRAM controller never “sees” the new cache. It just handles memory read/write requests as normal. The same goes for the Arm processors. They operate normally. No extra cache coherence hardware or software is required. Like all good caches, this last-level cache is transparent to software. 

 

### The Pros and Cons of a Last-Level Cache

- Like all good caches, a last-level cache helps improve performance dramatically without resorting to exotic or expensive (or ill-defined) memory technologies. It makes the generic DRAMs you have work better. Specifically, it improves both latency and bandwidth, because the cache is on-chip and far faster than off-chip DRAM, and because it has a wider, faster connection to the CPU cluster. It’s a win-win-win. 

- A cache takes up die area, of course. The cache-control logic (managing tags, lookup tables, etc.) is negligible, but the cache RAM itself uses a measurable amount of space. On the other hand, a last-level cache saves on power consumption, because nothing consumes more energy than reading or writing to/from external DRAM, especially when you’re hammering DRAM the way modern workloads do. Every transaction that stays on-chip saves a ton of power as well as time. 

- There’s also a security benefit. On-chip cache transactions can’t be snooped or subjected to side-channel attacks from signal probing or RF manipulation. Caches are tested, well-understood technology. That doesn’t mean they’re simple or trivial to implement – there’s a real art to designing a good cache – but at least you know you’re not beta-testing some vendor’s proprietary bleeding-edge memory interface.