import torch
import torch.nn as nn

def Batchnorm_simple_for_train(x, gamma, beta, bn_param):
    '''
    param:x 输入数据，设shape(B,L)
    param:gama 缩放因子
    param:beta 平移因子
    param:bn_param batch_norm需要的一些参数
    eps: 接近0的参数，防止分母出现0
    momentum：动量参数，一般为0.9， 0.99， 0.999
    running_mean: 滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
    running_var:滑动平均的计算方式计算方差，训练时计算，为测试数据做准备
    '''
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    results = 0. #建立一个新的变量

    x_mean = x.mean(axis=0) # Calculate mean of x
    x_var = x.var(axis=0) # Calculate variance of x
    x_normalized = (x-x_mean)/np.sqrt(x_var + eps)  # Normalization
    results = gamma * x_normalized + beta

    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    # Record the new value
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return results, bn_param

def Batchnorm_simple_for_test(x, gamma, beta, bn_param):

    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    results = 0.

    x_normalized = (x - running_mean)/np.sqrt(running_var + eps)
    results = gamma * x_normalized + beta

    return results, bn_param