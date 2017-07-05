from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
def basic_block(bottom,k3x3_bin_out,k1x1_out,name="",lr=1):
    if name=="":
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                                       param=[dict(lr_mult=1, decay_mult=1)],
                                       num_output=k1x1_out, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                       bias_filler=dict(type='constant'))
    else:
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                         param=[dict(name=name,lr_mult=lr, decay_mult=0)],
                    num_output=k1x1_out, pad=0, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_input_1x1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))


    relu_act = L.ReLU(bn_act, in_place=True)
    from_bottom_3x3_bin =  L.Convolution(relu_act, kernel_size=3,
                         param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                    num_output=k3x3_bin_out, pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_bottom_3x3_bin, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                         scale_param=dict(bias_term=True,filler=dict(value=1),bias_filler=dict(value=0)))
    #relu_act = L.ReLU(bn_act, in_place=True)
    return bn_act #relu_act

def basic_block2(bottom,k3x3_bin_out,k1x1_out,name="",lr=1):
    if name=="":
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                                       param=[dict(lr_mult=1, decay_mult=1)],
                                       num_output=32, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                       bias_filler=dict(type='constant'))
    else:
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                         param=[dict(name=name,lr_mult=lr, decay_mult=1)],
                    num_output=32, pad=0, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_input_1x1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))

    #relu_act = L.ReLU(bn_act, in_place=True)
    concat_act = L.Concat(bn_act,bottom,axis=1)
    from_1x1_bin = L.Convolution(concat_act, kernel_size=1, stride=1,
                                   param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                                   num_output=k1x1_out, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                   bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_1x1_bin, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
    relu_act = L.ReLU(bn_act, in_place=True)

    from_bottom_3x3_bin =  L.Convolution(relu_act, kernel_size=3,
                         param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                    num_output=k3x3_bin_out, pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_bottom_3x3_bin, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                         scale_param=dict(bias_term=True,filler=dict(value=1),bias_filler=dict(value=0)))
    relu_act = L.ReLU(bn_act, in_place=True)
    return relu_act
def basic_block3(bottom,k3x3_bin_out,k1x1_out,name="",lr=1):
    if name=="":
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                                       param=[dict(lr_mult=1, decay_mult=1)],
                                       num_output=32, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                       bias_filler=dict(type='constant'))
    else:
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                         param=[dict(name=name,lr_mult=lr, decay_mult=1)],
                    num_output=32, pad=0, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_input_1x1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))

    relu_act_1 = L.ReLU(bn_act, in_place=True)
    concat_act = L.Concat(relu_act_1,bottom,axis=1)
    from_1x1_bin = L.Convolution(concat_act, kernel_size=1, stride=1,
                                   param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                                   num_output=k1x1_out-32, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                   bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_1x1_bin, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
    relu_act_2 = L.ReLU(bn_act, in_place=True)
    concat_act = L.Concat(relu_act_1, relu_act_2, axis=1)
    from_bottom_3x3_bin =  L.Convolution(concat_act, kernel_size=3,
                         param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                    num_output=k3x3_bin_out, pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_bottom_3x3_bin, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                         scale_param=dict(bias_term=True,filler=dict(value=1),bias_filler=dict(value=0)))
    relu_act = L.ReLU(bn_act, in_place=True)
    return relu_act

def basic_block4(bottom,k3x3_bin_out,k1x1_out,name="",lr=1):
    if name=="":
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                                       param=[dict(lr_mult=1, decay_mult=1)],
                                       num_output=32, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                       bias_filler=dict(type='constant'))
    else:
        from_input_1x1 = L.Convolution(bottom, kernel_size=1, stride=1,
                         param=[dict(name=name,lr_mult=lr, decay_mult=1)],
                    num_output=32, pad=0, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_input_1x1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))

    relu_act_1 = L.ReLU(bn_act, in_place=True)
    concat_act = L.Concat(relu_act_1,bottom,axis=1)
    from_1x1_bin = L.Convolution(concat_act, kernel_size=1, stride=1,
                                   param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                                   num_output=k1x1_out-32, pad=0, bias_term=False, weight_filler=dict(type='msra'),
                                   bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_1x1_bin, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
    relu_act_2 = L.ReLU(bn_act, in_place=True)
    concat_act = L.Concat(relu_act_1, relu_act_2, axis=1)
    from_bottom_3x3_bin =  L.Convolution(concat_act, kernel_size=3,
                         param=[dict(lr_mult=1, decay_mult=1)],convolution_param=dict(is_binarized_param=True),
                    num_output=k3x3_bin_out, pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(from_bottom_3x3_bin, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                         scale_param=dict(bias_term=True,filler=dict(value=1),bias_filler=dict(value=0)))
    #relu_act = L.ReLU(bn_act, in_place=True)
    return bn_act
def block(bottom,nout,name="",lr=1):
    #b1 = basic_block(bottom,nout,nout,name,lr)
    #b2 = basic_block(b1,nout,nout,name,lr)
    #b1 = basic_block2(bottom,nout,nout,name,lr)
    #b2 = basic_block2(b1,nout,nout,name,lr)
    # b1 = basic_block3(bottom,nout,nout,name,lr)
    # b2 = basic_block3(b1,nout,nout,name,lr)
    # out = L.Eltwise(b2,bottom)

    # # combine binary and float
    # b1 = basic_block4(bottom,nout,nout,name,lr)
    # out_pre = L.Eltwise(b1, bottom)
    # relu_act = L.ReLU(out_pre,in_place=True)
    # b2 = basic_block4(relu_act,nout,nout,name,lr)
    # out = L.Eltwise(b2,relu_act)
    # relu_act = L.ReLU(out,in_place=True)

    # share float param
    b1 = basic_block(bottom,nout,nout,name,lr)
    out_pre = L.Eltwise(b1, bottom)
    relu_act = L.ReLU(out_pre,in_place=True)
    b2 = basic_block(relu_act,nout,nout,name,lr)
    out = L.Eltwise(b2,relu_act)
    relu_act = L.ReLU(out,in_place=True)

    return relu_act

def transition_bin(bottom, k3x3_out):
    bin_pool = L.Convolution(bottom, kernel_size=3, stride=2,
                         param=[dict(lr_mult=1, decay_mult=1)],
                             convolution_param=dict(is_binarized_param=True),
                    num_output=k3x3_out, pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    bn_act = L.BatchNormTorch(bin_pool, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
    relu_act = L.ReLU(bn_act, in_place=True)
    return relu_act

def bin_net(data_file,batch_size=64,depth=[4,4,4,4],first_output=64,out_dim=[64,128,256,512]):
    data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                         transform_param=dict(mean_file="/home/zl499/caffe/examples/cifar10/mean.binaryproto"))

    nchannels = first_output
    model = L.Convolution(data, kernel_size=7, stride=2, num_output=nchannels,
                          param=[dict(lr_mult=1, decay_mult=1)],
                          pad=3, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    model = L.BatchNormTorch(model, in_place=False,
                             param=[dict(lr_mult=0, decay_mult=0),
                                    dict(lr_mult=0, decay_mult=0),
                                    dict(lr_mult=0, decay_mult=0),
                                    dict(lr_mult=1, decay_mult=0),
                                    dict(lr_mult=1, decay_mult=0)],
                             scale_param=dict(bias_term=True, filler=dict(value=1),
                                              bias_filler=dict(value=0)))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1, ceil_mode=False)  # global_pooling=True)


    for idx,d in enumerate(depth):
        k3x3_bin = out_dim[idx]
        #name = "" #
        name = 'param_b%d'%idx
        #lr = 1/d

        for i in range(d/2):
            model = block(model,k3x3_bin,name)
        if idx<(len(depth)-1):
            model = transition_bin(model,out_dim[idx+1])

    model = L.Pooling(model, pool=P.Pooling.AVE, kernel_size=7, stride=1)#global_pooling=True)
    model = L.InnerProduct(model, num_output=1000, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'),
                           param=[dict(lr_mult=1, decay_mult=1),
                                  dict(lr_mult=1, decay_mult=0)])
    # model = L.Convolution(model, num_output=1000, bias_term=True, weight_filler=dict(type='msra'),kernel_size=1, stride=1,pad=0,
    #                       param=[dict(lr_mult=1, decay_mult=1),
    #                              dict(lr_mult=1, decay_mult=0)],
    #                        bias_filler=dict(type='constant'))
    loss = L.SoftmaxWithLoss(model, label)
    accuracy = L.Accuracy(model, label)

    return to_proto(loss,accuracy)

def make_net():

    # with open('DesNet121.prototxt', 'w') as f:
    #     #change the path to your data. If it's not lmdb format, also change first line of densenet() function
    #     print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', batch_size=64)), file=f)
    #with open('ResNet34_share_bin.prototxt', 'w') as f:
    #with open('ResNet34_nl1x1_3.prototxt', 'w') as f:
    with open('ResNet34_share_bin2.prototxt', 'w') as f:
        #change the path to your data. If it's not lmdb format, also change first line of densenet() function
        #print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', batch_size=64,depth=[6,12,36,24], growth_rate=48,first_output=96)), file=f)
        print(str(
            bin_net('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb',depth=[6,8,12,6])), file=f)
            #bin_net('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', depth=[4, 4, 4, 4])), file=f)

    # with open('test_densenet.prototxt', 'w') as f:
    #     print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_test_lmdb', batch_size=50)), file=f)

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_densenet.prototxt'
    s.test_net.append('test_densenet.prototxt')
    s.test_interval = 800
    s.test_iter.append(200)

    s.max_iter = 230000
    s.type = 'Nesterov'
    s.display = 1

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':

    make_net()
    #make_solver()










