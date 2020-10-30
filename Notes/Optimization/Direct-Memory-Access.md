### Introduction
- DMA terchnique is used to rransfer large volumes of data between IO interfaces and the memory
- Also used for 
   - intra chips data transfer in multi core processors
   - memory to memory copying or moving of data
- A hardware controller  konw as DMA Controller is uded that can act a processig unit by genereating addressed and initiating memort read and write cycles.
- The CPU configures the DMAC and delegates the IO operations to it.
###### Overview
- Whenever request for data transfer, the IO interfce signals the DMAc
- DMAC requst the system bus from the CPU
- The CPU complestes the current bus scycle and isolates itself from the sysyten bus and  responsed to the DMA by activating BG
- Now thre DMAC is responsible for generaitng all bus signals and performing the transfer
- The CPPU can perform its internal operations
- The dat a dont pass through the CPU but the system bus is ocupied.
![](./resources/dma2.JPG)
###### DMA Terms
- Event - a hardware signal that initiates a transfer on a DMA channel
- Channel - one thread of transfer. It contains source and destination addresses plus control information
- Element: 8-, 16-, 32-bit datum
- Frame: Group of elements
- Array: Group of contiguous elements
- Block: Group of Frames or Arrays
###### DMA vs Cache
- cache has problems in some DSP applications:
   - Data isn’t always reused, or has limited reuse
   - Cache only loads data when requested the first time
   - Cache line size may not be a good match for data size

### DMA transfer modes

######
######
######
######
######
######
######
######
######
######
######
######