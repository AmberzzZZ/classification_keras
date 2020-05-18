### vanilla version  with bypass_mode==None
    激活函数都是relu
    2016年的paper，还没引入BN层，有一层Dropout
    3个maxpooling和开头一个stride2 conv，output_stride=16


### simple bypass with bypass_mode=='simple'
    bypass有助于缓解bottleneck问题
    simple model claims the highest acc


### complex bypass with bypass_mode=='complex'
    1x1 conv on id path