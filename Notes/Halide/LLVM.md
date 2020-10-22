

### Architecture

![](./resources/llvm.png)



### Frontend
- libclangLex 
  - This library is used for pre-processing and lexical analysis, handling macro and pragma constructions

- libclangAST
  - all abstract syntax tree related functionalitites.

- libclangparse
  - used for parsing logic using the results from the lexical phase

- libclangSema
   - This library is used for semantic analysis, used for AST verification.

- libclanfCodegen
   - LLVM IR code generator using target specific information

- libclangAnalysis
   - This library contains the resources for static analysis

- libclangRewrite
   - This library allows support for code rewriting and proving to build code refactoring tools

- libclangBasic
   - This library provides a set of utilities - memory allocation abstractions, source locations, and diagnostics.



##### LLVM IR 

- a low -level programming language similar to assembly
- strongly typed RISC instruction set which abstracts away most details of the target.
- No fixed set of registers.
- Has three forms :
  - a human-readable format
  - an in-memory format for frontends
  - a dense bit code for serializing
  
