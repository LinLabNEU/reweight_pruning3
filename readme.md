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



# Quantization Method

All layers of this current model are quantized into 8 bits. We use linear quantization implmented at [here](https://github.com/LinLabNEU/reweight_pruning3/blob/master/quantization/quant.py#L56) to quantize pruned model.

To run the quantization,
```
python ./quantization/main.py
```
or refer to the ./quantization/quantization.sh

1. How to implement linear quantization?

Our implementation refers to [link](https://github.com/LinLabNEU/reweighted_prune4/blob/master/quantization/quant.py).
We implement linear quantization in [quant.py] (https://github.com/LinLabNEU/reweighted_prune4/blob/master/quantization/quant.py) according to Equation Xq[k] = round( delta * floor(X[k] / delta + 1/2 ) ), where X[k] is one element in inputs X, delta is the step size, and round() function bounds the quantized result into range [-math.pow(2.0, bits-1), math.pow(2.0, bits-1)-1]. The main code is shown as following, and all the codes can be found in [quant.py](https://github.com/LinLabNEU/reweighted_prune4/blob/master/quantization/quant.py) file.

```python
def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value
```

2. How to apply layer-wise quantization into pruned model?

Firstly, we load the pruned model and read all the weights in each layer using model.state_dict. Then we perform linear quantizization on each layer's weights using predefined Quantization Bits Number(QBNs) Vector as following code shows.

```python
QBNs=setQBN(LayerSize,1)
#all layers are quantized into 8bits.
    count=-1
    for k, v in state_dict.items():
        sizes=len(list(v.size()))
        if sizes > 1:#removing bias
            #channels=list(v.size())[0]
            count=count+1                
            if QBNs[count]<32:
                bits = QBNs[count]
                if args.quant_method == 'linear':
                    sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=args.overflow_rate)
                    v_quant  = quant.linear_quantize(v, sf, bits=bits)
                elif args.quant_method == 'log':
                    v_quant= quant.log_minmax_quantize(v, bits=bits)
                elif args.quant_method == 'minmax':
                    v_quant= quant.min_max_quantize(v, bits=bits)
                else:
                    v_quant = quant.tanh_quantize(v, bits=bits)
            else:
                v_quant=v
        else:
            v_quant=v
        state_dict_quant[k] = v_quant
    model.load_state_dict(state_dict_quant)
```

Then we quantize all the inputs like activations, batchnorms, poolings using predefined QBNs, as following shows:

```python
 #quantize forward activation
    if args.fwd_bits < 32:
        model_new = quant.duplicate_model_with_quant(model, bits=args.fwd_bits, overflow_rate=args.overflow_rate,
                                                 counter=args.n_sample, type=args.quant_method)
        model.load_state_dict(model_new.state_dict())#print(model_new)
```

3. Avoiding accumulation overflow

Like [this post](https://nervanasystems.github.io/distiller/quantization.html) said, convolution and fully connected layers involve the storing of intermediate results in accumulators. Due to the limited dynamic range of integer formats, if we would use the same bit-width for the weights and activation, and for the accumulators, we would likely overflow very quickly. Therefore, accumulators are usually implemented with higher bit-widths. The result of multiplying two n-bit integers is, at most, a 2n-bit number. In convolution layers, such multiplications are accumulated c⋅k^2 times, where c is the number of input channels and k is the kernel width (assuming a square kernel). Hence, to avoid overflowing, the accumulator should be 2n+M-bits wide, where M is at least log2(c⋅k2). In our case, all the multiplication in the quantized MobileNet2 is not overflow. Since for a convolution operation, element-wise multiplication is performed firstly, and each element in the kernel and each element in the input are quantized into n-bits, for example 8bits, then each multiplication in the convolution is just 8-bits multiplication, the product is 2n-bits which leads to the first-time accumulation will be at least 2n-bits without rounding the product back into 8bits. The first-time accumulation between each two numbers occupies 50% of all additions in the convolution. Therefore half of the addition is just 2n-bits, the 25% addition (The second accumulation) will be (2n+1) bits, the 12.25% addition (The third accumulation) will be (2n+2) bits, this action will be repeated until addtion is 32 bits or all the accumulations are done. This is a so-called "binary-tree" accumulation, which decide how we caculate the operation number of additions in this micronet challenge. 

4. Testing and result

All the inputs and weights of 57 convolutional layers and 1 layer Fully Connected layers are quantized into 8 bits.
With 8 bits for each layer, we can achieve 80.18% accuracy. And the number bits of multiplications in the convolutional layers and Fully connected layers is the quantized bit, e.g. n-bit, but the number bits of accumulation(addition) is mixed precision in order to avoid overflow.(50% 2n bits, 25% (2n+1) bits, 12.5% (2n+2) bits until to 32 bits or all the accumulations in one convolution are done.)

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
We first count the number of additions and multiplications by setting the addition bits and multiplication bits to 32 in the check_model_operations.py. It shows that the there are 155.692M multiplications and 153.41 additions in the case of no sparsity (setting the sparsity to 0 when print_summary). The total number of operations is 309M, which is close to and no larger than the 325.4M value in the first counting method with the tool.

In the pruned model, we should set the sparsity to a non-zero value, and the number of operations will decrease. But since the sparsity for each layer is not the same, it is not easy to use one number to represent the sparsity of all layers.  We think that setting the sparsity parameter to 0.5 should be an appropriate choice, considering the overall sparsity for the whole model is about 75% (0.98M / 4M). By setting the sparsity parameter to 0.5, there are 78.99M multiplications and 76.7M additions according to the outputs of the check_model_operations.py file (still setting the addition bits and multiplication bits to 32 bits). 

We perform quantization for this model and all of the layers are quantized to 8bits. As specified in the Quantization part, the multiplication bits after quantization should the the same as the quantization bits, we set the multiplication bits to 8 and count the multiplication as 32 bits. So the multiplication number is 19.75M (78.99M * 8 / 32). 
For the addition, 50% are 16bits, 25% are 17bits, 12.5% are 18bits, 6.25% are 19bits, 3.125% are 20bits and so on. For the simpliclity, we set the rest 3.125% addition to 32bits. So the total number of addition is 41.5M
(76.7M * (0.5 * 16 + 0.25 * 17 + 0.125 * 18 + 0.0625 * 19 + 0.03125 * 20+0.03125 * 32) / 32). Then the total number of operations for scoring is 61.25M (19.75M + 41.5M).

- Operation number: 61.25M

# Score 

For CIFAR-100, parameter storage and compute requirements will be normalized relative to WideResNet-28-10, which has 36.5M parameters and 10.49B math operations.

So the score is 0.3671M / 36.5M + 61.25M / 10.49B = 0.0159.


# Team member

The team name is Woody.

This is an collaboration of Northeastern University, Indiana University and MIT-IBM Watson AI Lab, IBM Research. The team members are listed as follows, 
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
- MIT-IBM Watson AI Lab, IBM Research
  - Gaoyuan Zhang
  - Sijia Liu

contact: zhao.pu@husky.neu.edu or zhan.zhe@husky.neu.edu or xue.lin@northeastern.edu
