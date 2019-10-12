# Reweighted pruning

This is the submission for the MicroNet challenge on the CIFAR-100 task. 

This project is based on Pytorch. We perform quantization for this model. We first introduce the pruning and quantization method for this project. Then we demonstrate how we count the parameters and operations and show the score in the end.

# Model

The model architecture is based on the MobileNet V2. For more details, please refer to the mobilenet_v2_cifar100_exp_30.py file and the original paper. The pruned model is saved as cifar100_mobilenetv217_retrained_acc_80.510mobilenetv217_quantized_acc_80.180_config_vgg16_threshold.pt. It can achieve 80.01% accuracy satisfying the 80% accuracy requirement.


# Pruning method

We use reweighted L1 pruning method to prune the model. The detailed method is shown in training/reweighted_l1_prune.pdf. The code for the pruning is in the training directory. Basically, starting from a pretrained unpruned model which achieves 81.92% accuracy on CIFAR-100, we first try to decrease the L1 norm of this model with the reweighted L1 pruning method to make model sparse. Then we set the parameters under a threshold to zero (obtain the sparsity mask) and retrain the model. Note that during retraining, the zero parameters are not updated.

To run the pruning:

```
python training/main.py
```

or refer to the training/run.sh file.



# Quantization Analysis

All layers of this current model are 8 bits. To run the quantization,
```
python quantization/main.py
```

1. All the multiplication in the quantized MobileNet2 is not overflow. Since for a convolution operation, element-wise multiplication is performed firstly, and each element in the kernel and each element in the input are quantized into Qbits, for example 8bits, then each multiplication in the convolution is just Qbits-multiplication, the product is 2Q-bits which leads to accumulation will be at least 2Q-bits without rounding the product back into 8bits. Therefore without rounding in the convolution, addition taking 2Q-bits product in the convolution will be overflow. And half of the addition is just 2Q-bits, the 25% addition will be (2Q+1) bits, the 12.25% will be (2Q+2) bits. This is a so-called "binary-tree" accumulation. 


# Verify model

To load and verify the model, run:

```
python testers.py
```
It outputs the test accuracy of the model. It also counts the number of non-zero and zero elements in the parameters. The total number of parameters is 3996704 (4.0M). Among them, there are 3012168 (3.02M) zero parameters and 984536 (0.98M) non-zero parameters. The number of bitmask is 122051 (0.1221M). So the total parameters for storage is 0.3671M (0.98M / 4 + 0.1221M) since the parameters are all 8bit.

# Count parameters

From the output of the testers file, that is,
```
python testers.py
```
It outputs the test accuracy of the model. It also counts the number of non-zero and zero elements in the parameters. The total number of parameters is 3996704 (4.0M). Among them, there are 3012168 (3.02M) zero parameters and 984536 (0.98M) non-zero parameters. The number of bitmask is 122051 (0.1221M). So the total parameters for storage is 0.3671M (0.98M * 8 / 32 + 0.1221M) since the parameters are all 8bit.

- Parameter number: 0.3671M

# Count operations

We show how we count the operations in this part. 

We tried two ways to count the operations. One way is to use the open-source [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter) tool. It will count the number of operations during inference. To use this tool and check the performance,

```
pip install thop
python mobilenetv2.py
```
It shows that the total number of operations is 325.4M. It counts the real operations during runtime and does not consider the sparsity since zero parameters still participate in the operations. We do not use this number for scoring and this number can work as a reference. 

We would like to use the second method to count the number of operations. It is based on the counting example from the MicroNet challenge group. ( https://github.com/google-research/google-research/blob/master/micronet_challenge/ )
The original version is for the efficientnet on tensorflow. We made necessary modifications to work for the our mobilenet_v2 model on pytorch. To run the counting,
```
python check_model_operations.py
```
We first count the number of additions and multiplications by setting the addition bits and multiplication bits to 32 in the check_model_operations.py. It shows that the there are 155.692M multiplications and 153.41 additions in the case of no sparsity (setting the sparsity to 0 when print_summary). The total number of operations is 309M, which is close to and no larger the 325.4M results in the first counting method with the tool.

In the pruned model, we should set the sparsity to a non-zero value, and the number of operations will decrease. But since the sparsity for each layer is not the same, it is not easy to use one number to represent the sparsity of all layers.  We think that setting the sparsity parameter to 0.5 should be an appropriate choice (not over-estimating the results), considering the overall sparsity for the whole model is about 75% (0.98M / 4M). By setting the sparsity parameter to 0.5, there are 78.99M multiplications and 76.7M additions according to the outputs of the check_model_operations.py file (still setting the addition bits and multiplication bits to 32 bits). 

We perform quantization for this model and all of the layers are quantized to 8bits. As specified in the Quantization part, the multiplication bits after quantization should the the same as the quantization bits, we set the multiplication bits to 8 bits and count the multiplication as 32 bits. So the multiplication number is 19.75M (78.99M * 8 / 32). 
For the addition, 50% are 16bits, 25% are 17bits, 12.5% are 18bits, 6.25% are 19bits, 3.125% are 20bits and so on. For the simpliclity, we set the rest 3.125% addition to 32bits. So the total number of addition is 41.5M
(76.7M * (0.5 * 16 + 0.25 * 17 + 0.125 * 18 + 0.0625 * 19 + 0.03125 * 20+0.03125 * 32) / 32). So the total number of operations for scoring is 61.25M (19.75M + 41.5M).

- Operation number: 61.25M

# Score 

For CIFAR-100, parameter storage and compute requirements will be normalized relative to WideResNet-28-10, which has 36.5M parameters and 10.49B math operations.

So the score is 0.3671M / 36.5M + 61.25M / 10.49B = 0.0159.


# Team member

The team name is Woody.

This is an collaboration of Northeastern University, Indiana University and IBM corporation. The team members are listed as follows, 
- Northeastern University
  - Pu Zhao
  - Zheng Zhan
  - Zhengang Li
  - Xiaolong Ma
  - Yanzhi Wang
  - Xue Lin
- Indiana University
  - Qian Lou
  - Lei Jiang
- IBM
  - Gaoyuan Zhang
  - Sijia Liu

contact: zhao.pu@husky.neu.edu or zhan.zhe@husky.neu.edu or xue.lin@northeastern.edu
