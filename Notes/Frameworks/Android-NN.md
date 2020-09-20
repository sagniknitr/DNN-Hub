## Introduction

- Supports callbacks for other tasks.
- Can preempt any process
- Supports back end learning.




## Hardware Abstraction Layer

#### Main data structure

####  . Request
- Refers to the input which is feed to or output which is extracted from, a NN model executed in the device.
- It also makes sure all the metadata regarding operands which are not updated during model compilation time.

  ```c++
  struct Request {

  }
  ```  
####  . RequestArgument
####  . DataLocation
####  . Operand
####  . OperandType