### Introduction
![](resources/quant1.JPG)
![](resources/quant2.png)
### Quantization aware training

In QAT, as opposed to computing scale factors to activation tensors after the DNN is trained, the quantization error is considered when training the model. The training graph is modified to simulate the lower precision behavior in the forward pass of the training process. This introduces the quantization errors as part of the training loss, which the optimizer tries to minimize during the training. Thus, QAT helps in modeling the quantization errors during training and mitigates its effects on the accuracy of the model at deployment.

However, the process of modifying the training graph to simulate lower precision behavior is intricate. You must modify the training code to insert FakeQuantization nodes for the weights of the DNN Layers and Quantize-Dequantize (QDQ) nodes to the intermediate activation tensors to compute their dynamic ranges.

The placement of these QDQ nodes is also critical so as not to affect the learning capabilities of the model. For example, we can combine the Conv -> Bias -> ReLU and Conv -> Bias -> BatchNormalization -> ReLU compute blocks. In that case, adding a QDQ layer only makes sense to the output tensor of the ReLU activation. For more information, see Quantization-Aware Training.

Not all activation layer types are friendly to quantization. For example, networks with regression layers typically require that the output tensors of these layers not be bound by the range of the 8-bit quantization and they may require finer granularity in representation than 8-bit quantization can provide. These layers work best if they are excluded from quantization.