# Reweighted pruning

This is the submission for the MicroNet challenge on the CIFAR-100 task. 

This project is based on Pytorch. We perform quantization for this model.

# Model

The model architecture is based on the MobileNet V2. For more details, please refer to the mobilenet_v2.py file and the original paper. The pruned model is saved as cifar100_mobilenetv2.pt. It can achieve 80.12% accuracy satisfying the 80% accuracy requirement.


total weights: 3905632, total number of zeros: 3005523, non-zeros: 900109, zero sparsity is: 0.7695\
only consider conv layers, compression rate is: 4.3391\

# Pruning method

We use reweighted L1 pruning method to prune the model. The detailed method is shown in XXX. The code for the pruning is in the training directory. Basically, starting from a pretrained unpruned model which achieves 81.92% accuracy on CIFAR-100, we first try to decrease the L1 norm of this model with the reweighted L1 pruning method to make model sparse. Then we set the parameters under a threshold to zero (obtain the sparsity mask) and retrain the model. Note that during retraining, the zero parameters are not updated.

To run the pruning:

```
python training/main.py
```

or refer to the training/run.sh file.



# Quantization Analysis

QBNs[3]=6bits\
QBNs[5]=6bits\
QBNs[8]=6bits\
QBNs[10]=6bits\
QBNs[11]=6bits\
QBNs[14]=6bits\
QBNs[16]=6bits\
QBNs[19]=6bits\
QBNs[Layers-3]=6bits\
Different from last model, above layers of this current model are 6 bits.

1. All the multiplication in the quantized MobileNet2 is not overflow. Since for a convolution operation, element-wise multiplication is performed firstly, and each element in the kernel and each element in the input are quantized into Qbits, for example 8bits, then each multiplication in the convolution is just Qbits-multiplication, the product is 2Q-bits which leads to accumulation will be at least 2Q-bits without rounding the product back into 8bits. Therefore without rounding in the convolution, addition taking 2Q-bits product in the convolution will be overflow. And half of the addition is just 2Q-bits, the 25% addition will be (2Q+1) bits, the 12.25% will be (2Q+2) bits. This is a so-called "binary-tree" accumulation. 


# Verify model

To load and verify the model, run:

```
python testers.py
```
It outputs the test accuracy of the model. It also counts the number of non-zero and zero elements in the parameters. The total number of parameters is XXX. Among them, there are XXXX zero parameters and XXXX non-zero parameters. 

# Count operations

We tried two ways to count the operations. One way is to use the open-source pytorch-opcounter tool. It will count the number of operations during inference. To use this tool and check the performance,
```
python XXXXX
```
It shows that the total number of operations is 325.4M. As it counts the real operations during runtime, and for unquantized models, multiplication are counted as 16bit while the operation counting is based on 32bit, we believe the operation number for scoring should be smaller than the value 325.4M. This number can work as a reference. 

We would like to use the second method to count the number of operations. It is based on the counting example from the MicroNet challenge group. ( https://github.com/google-research/google-research/blob/master/micronet_challenge/ )
The original version is for the efficientnet on tensorflow. We made necessary modifications to work for the our mobilenet_v2 model on pytorch. To run the counting,
```
python check_model_operations.py
```
It shows that the there are 77.84M multiplications and 153.41 additions in the case of no sparsity (setting the sparsity to 0 when print_summary). Since the multiplication is performed as 16bit and counted as 32bit, the actual number of multiplication should be 155.68M and the total number of operations is 309M, which is close to and no larger the 325.4M results in the first counting method with the tool.

So in the case of no sparsity, the total number of operations is 231.25M (77.84M+153.41M). If we consider the sparsity and set it to non-zero value, the number of operations will continue to reduce. But since the sparsity for each layer is not the same, it is hard to use one number to represent the sparsity of all layers. We would work on that if time permits. But if we do not have enough time, we think that we can use the number 231.25M to compute the score, although the real value should be smaller. 

# Score 

For CIFAR-100, parameter storage and compute requirements will be normalized relative to WideResNet-28-10, which has 36.5M parameters and 10.49B math operations.

So the score is XXX/36.5M + 231.25M/10.49B = XXX.
