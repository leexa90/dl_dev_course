Python 2.7.13 |Anaconda, Inc.| (default, Sep 30 2017, 18:12:43) 
[GCC 7.2.0] on linux2
Type "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
737990
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 297, in <module>
    sorted_data = sorted(zip(X_train,y_train),key = lambda x : x[0][-1][0])
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1306, in _run_fn
    status, run_metadata)
KeyboardInterrupt
>>> tf.layers.max_pooling1d(layer3_DO, (2,1),strides = (2,1))

Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    tf.layers.max_pooling1d(layer3_DO, (2,1),strides = (2,1))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 224, in max_pooling1d
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 194, in __init__
    **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 66, in __init__
    self.pool_size = utils.normalize_tuple(pool_size, 1, 'pool_size')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/utils.py", line 84, in normalize_tuple
    str(n) + ' integers. Received: ' + str(value))
ValueError: The `pool_size` argument must be a tuple of 1 integers. Received: (2, 1)
>>> tf.layers.max_pooling1d(layer3_DO, 2,strides = (2,1))

Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    tf.layers.max_pooling1d(layer3_DO, 2,strides = (2,1))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 224, in max_pooling1d
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 194, in __init__
    **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 67, in __init__
    self.strides = utils.normalize_tuple(strides, 1, 'strides')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/utils.py", line 84, in normalize_tuple
    str(n) + ' integers. Received: ' + str(value))
ValueError: The `strides` argument must be a tuple of 1 integers. Received: (2, 1)
>>> tf.layers.max_pooling1d(layer3_DO,2,(2,1))

Traceback (most recent call last):
  File "<pyshell#2>", line 1, in <module>
    tf.layers.max_pooling1d(layer3_DO,2,(2,1))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 224, in max_pooling1d
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 194, in __init__
    **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 67, in __init__
    self.strides = utils.normalize_tuple(strides, 1, 'strides')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/utils.py", line 84, in normalize_tuple
    str(n) + ' integers. Received: ' + str(value))
ValueError: The `strides` argument must be a tuple of 1 integers. Received: (2, 1)
>>> Traceback (most recent call last):
KeyboardInterrupt
>>> tf.layers.max_pooling1d(layer3_DO,2,1)

Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    tf.layers.max_pooling1d(layer3_DO,2,1)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 225, in max_pooling1d
    return layer.apply(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 503, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 437, in __call__
    self._assert_input_compatibility(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 551, in _assert_input_compatibility
    str(x.get_shape().as_list()))
ValueError: Input 0 of layer max_pooling1d_4 is incompatible with the layer: expected ndim=3, found ndim=4. Full shape received: [None, 1, None, 256]
>>> tf.layers.max_pooling1d(layer3_DO,2,2)

Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    tf.layers.max_pooling1d(layer3_DO,2,2)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 225, in max_pooling1d
    return layer.apply(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 503, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 437, in __call__
    self._assert_input_compatibility(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 551, in _assert_input_compatibility
    str(x.get_shape().as_list()))
ValueError: Input 0 of layer max_pooling1d_5 is incompatible with the layer: expected ndim=3, found ndim=4. Full shape received: [None, 1, None, 256]
>>> tf.layers.max_pooling2d(layer3_DO,2,2)

Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    tf.layers.max_pooling2d(layer3_DO,2,2)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 426, in max_pooling2d
    return layer.apply(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 503, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 450, in __call__
    outputs = self.call(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 276, in call
    data_format=utils.convert_data_format(self.data_format, 4))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_ops.py", line 1772, in max_pool
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 1605, in _max_pool
    data_format=data_format, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2632, in create_op
    set_shapes_for_outputs(ret)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1911, in set_shapes_for_outputs
    shapes = shape_func(op)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1861, in call_with_requiring
    return call_cpp_shape_fn(op, require_shape_fn=True)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 595, in call_cpp_shape_fn
    require_shape_fn)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 659, in _call_cpp_shape_fn_impl
    raise ValueError(err.message)
ValueError: Negative dimension size caused by subtracting 2 from 1 for 'max_pooling2d/MaxPool' (op: 'MaxPool') with input shapes: [?,1,?,256].
>>> tf.layers.max_pooling2d(layer3_DO,(1,2),2)
<tf.Tensor 'max_pooling2d_2/MaxPool:0' shape=(?, 1, ?, 256) dtype=float32>
>>> tf.layers.max_pooling2d(layer3_DO,(1,2),(1,2))
<tf.Tensor 'max_pooling2d_3/MaxPool:0' shape=(?, 1, ?, 256) dtype=float32>
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 99, in <module>
    for j in  range(len(i)):
KeyboardInterrupt
>>> tf.layers.max_pooling2d(layer3_DO,(1,2),(1,2))

Traceback (most recent call last):
  File "<pyshell#8>", line 1, in <module>
    tf.layers.max_pooling2d(layer3_DO,(1,2),(1,2))
NameError: name 'layer3_DO' is not defined
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 164, in <module>
    layer3_DO = tf.layers.max_pooling1d(layer3_DO, (2,1),strides = (2,1))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 224, in max_pooling1d
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 194, in __init__
    **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 66, in __init__
    self.pool_size = utils.normalize_tuple(pool_size, 1, 'pool_size')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/utils.py", line 84, in normalize_tuple
    str(n) + ' integers. Received: ' + str(value))
ValueError: The `pool_size` argument must be a tuple of 1 integers. Received: (2, 1)
>>> tf.layers.max_pooling2d(layer3_DO,(1,2),(1,2))
<tf.Tensor 'max_pooling2d/MaxPool:0' shape=(?, 1, 2, 256) dtype=float32>
>>> tf.layers.max_pooling2d(layer3_DO,(1,2),(1,2))
<tf.Tensor 'max_pooling2d_2/MaxPool:0' shape=(?, 1, 2, 256) dtype=float32>
>>> tf.layers.max_pooling2d(layer3_DO,(2,2),(1,2))

Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    tf.layers.max_pooling2d(layer3_DO,(2,2),(1,2))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 426, in max_pooling2d
    return layer.apply(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 503, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 450, in __call__
    outputs = self.call(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 276, in call
    data_format=utils.convert_data_format(self.data_format, 4))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_ops.py", line 1772, in max_pool
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 1605, in _max_pool
    data_format=data_format, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2632, in create_op
    set_shapes_for_outputs(ret)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1911, in set_shapes_for_outputs
    shapes = shape_func(op)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1861, in call_with_requiring
    return call_cpp_shape_fn(op, require_shape_fn=True)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 595, in call_cpp_shape_fn
    require_shape_fn)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 659, in _call_cpp_shape_fn_impl
    raise ValueError(err.message)
ValueError: Negative dimension size caused by subtracting 2 from 1 for 'max_pooling2d_3/MaxPool' (op: 'MaxPool') with input shapes: [?,1,5,256].
>>> tf.layers.max_pooling2d(layer3_DO,(1,2),(1,2))
<tf.Tensor 'max_pooling2d_4/MaxPool:0' shape=(?, 1, 2, 256) dtype=float32>
>>> tf.layers.max_pooling2d(layer3_DO,(2,2),(1,2))

Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    tf.layers.max_pooling2d(layer3_DO,(2,2),(1,2))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 426, in max_pooling2d
    return layer.apply(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 503, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 450, in __call__
    outputs = self.call(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 276, in call
    data_format=utils.convert_data_format(self.data_format, 4))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_ops.py", line 1772, in max_pool
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 1605, in _max_pool
    data_format=data_format, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2632, in create_op
    set_shapes_for_outputs(ret)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1911, in set_shapes_for_outputs
    shapes = shape_func(op)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1861, in call_with_requiring
    return call_cpp_shape_fn(op, require_shape_fn=True)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 595, in call_cpp_shape_fn
    require_shape_fn)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 659, in _call_cpp_shape_fn_impl
    raise ValueError(err.message)
ValueError: Negative dimension size caused by subtracting 2 from 1 for 'max_pooling2d_5/MaxPool' (op: 'MaxPool') with input shapes: [?,1,5,256].
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 164, in <module>
    layer3_pool = tf.layers.max_pooling2d(layer3_pool, (2,1),strides = (2,1),name='pool1')
NameError: name 'layer3_pool' is not defined
>>> 
KeyboardInterrupt
>>> tf.layers.max_pooling2d(layer3_DO, (2,1),strides = (2,1),name='pool1')

Traceback (most recent call last):
  File "<pyshell#14>", line 1, in <module>
    tf.layers.max_pooling2d(layer3_DO, (2,1),strides = (2,1),name='pool1')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 426, in max_pooling2d
    return layer.apply(inputs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 503, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/base.py", line 450, in __call__
    outputs = self.call(inputs, *args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/layers/pooling.py", line 276, in call
    data_format=utils.convert_data_format(self.data_format, 4))
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_ops.py", line 1772, in max_pool
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 1605, in _max_pool
    data_format=data_format, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2632, in create_op
    set_shapes_for_outputs(ret)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1911, in set_shapes_for_outputs
    shapes = shape_func(op)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1861, in call_with_requiring
    return call_cpp_shape_fn(op, require_shape_fn=True)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 595, in call_cpp_shape_fn
    require_shape_fn)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/common_shapes.py", line 659, in _call_cpp_shape_fn_impl
    raise ValueError(err.message)
ValueError: Negative dimension size caused by subtracting 2 from 1 for 'pool1/MaxPool' (op: 'MaxPool') with input shapes: [?,1,?,256].
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
737990
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 295, in <module>
    dropout : 0.0,learning_rate : lr}) #sgd
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1306, in _run_fn
    status, run_metadata)
KeyboardInterrupt
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
737990
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.75891 0.751285
0.90194833842 0.871010224992 0.845582655827
0.872173 0.866906
0.936672698953 0.889173940757 0.864943089431
0.872858 0.86074
0.956488609183 0.907063725774 0.884975609756
SAVED

0.901474 0.877698
0.974290756197 0.918236939561 0.899609756098
SAVED

0.904901 0.882837
0.980440987134 0.90972928267 0.893273712737
SAVED

0.940884 0.925488
0.985785712785 0.922909565978 0.899853658537
SAVED

0.95425 0.936793
0.989384076565 0.921466161651 0.895327913279
SAVED

0.956134 0.933196
0.992692530469 0.919480163726 0.906319783198
SAVED

0.965216 0.947071
0.9942873349 0.923473231171 0.893826558266
SAVED

0.958019 0.935766
0.99611382714 0.916714516749 0.906823848238
0.976868 0.952724
0.996624620731 0.915897992404 0.895062330623
0.973441 0.945529
0.997511756956 0.912394839566 0.893046070461

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 312, in <module>
    labels: labels_,dropout : 0,learning_rate : lr})
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1306, in _run_fn
    status, run_metadata)
KeyboardInterrupt
>>> tf.keras

Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    tf.keras
AttributeError: 'module' object has no attribute 'keras'
>>> tf.layers.max_pooling1d(AttributeError: 'module' object has no attribute 'keras'
KeyboardInterrupt
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
737990
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.771761 0.760021
0.916511057691 0.868976815976 0.839100271003
0.89719 0.884378
0.951147184783 0.893014239131 0.866401084011

 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869062
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.758739 0.755396
0.905106735453 0.878975288286 0.826742547425
0.898389 0.8926
0.947826726329 0.887356515601 0.855680216802
0.912783 0.897225
0.972256585246 0.91787345453 0.88308401084
SAVED

0.936086 0.912127
0.985749098908 0.922788404301 0.887875338753
SAVED

0.947224 0.92446
0.990765800238 0.92670772116 0.891593495935
SAVED

0.967101 0.943988
0.994963191049 0.927787640455 0.900249322493
SAVED

0.978067 0.951696
0.996419342931 0.933324202308 0.877761517615
SAVED

0.972755 0.953751
0.997839481163 0.930179266603 0.883539295393
SAVED

0.974297 0.947585
0.998457115247 0.925164226751 0.897685636856
0.98218 0.955807
0.99873321989 0.932950181479 0.904910569106
SAVED

0.988348 0.961973
0.999403073765 0.917694345964 0.877739837398
0.986292 0.959404
0.996703250532 0.936690389772 0.891479674797
SAVED

0.986806 0.960432
0.999320842599 0.932797412408 0.889241192412
0.984407 0.953238
0.999278226448 0.940383186974 0.890818428184
SAVED

0.991432 0.963001
0.999551329954 0.928888631347 0.895615176152
0.993317 0.967112
0.999678578181 0.933002860469 0.911311653117
0.989719 0.96557
0.999598147698 0.936200475164 0.902016260163
SAVED

0.996059 0.96814
0.999811828683 0.930600698523 0.900829268293
0.99366 0.96557
0.9997073891 0.923317828151 0.885008130081
0.995545 0.96814
0.999775815034 0.931817583193 0.883398373984
0.995888 0.96814
0.999930673725 0.923615464444 0.886032520325
0.997258 0.969681
0.999967287602 0.926194101007 0.888113821138
0.997772 0.970195
0.999947480095 0.923465329323 0.890672086721
0.998286 0.96814
0.999945079185 0.923022825806 0.890715447154
0.997944 0.96814
0.999961285327 0.925095744064 0.891577235772
0.998629 0.973792
0.999920469858 0.918489798714 0.885983739837
0.994345 0.963001
0.999797723337 0.938528886524 0.894357723577
SAVED

0.996744 0.969681
0.999809427773 0.938481475433 0.897815718157
SAVED

0.992118 0.963001
0.999790820721 0.942637847747 0.900140921409
SAVED

0.996744 0.968654
0.999818431185 0.930748199696 0.899783197832
0.998458 0.970195
0.999823833233 0.925564587076 0.890699186992
0.998801 0.971737
0.999952281915 0.929009793024 0.892390243902
0.998801 0.970195
0.999984694199 0.925216905741 0.897116531165
0.999143 0.971737
0.999852043925 0.922983316564 0.877891598916
0.86035 0.859712
0.918570438196 0.879928778005 0.846444444444
0.819397 0.813464
0.947030224455 0.88804661037 0.872276422764
0.904044 0.890031
0.968509965277 0.908575612788 0.879907859079
0.924092 0.90853
0.988702818368 0.909697675276 0.896601626016
0.959219 0.93628
0.985348146948 0.898071422175 0.870287262873
0.969157 0.945015
0.995646249929 0.923952609981 0.876281842818
0.9695 0.945529
0.996199059444 0.933139825843 0.902075880759
0.971213 0.946043
0.997431326473 0.917357200428 0.899701897019
0.982522 0.961459
0.998657591227 0.92760326399 0.892319783198
0.988348 0.967626
0.998821453331 0.925554051278 0.897951219512
0.988177 0.966598
0.998922891776 0.927408351727 0.894558265583
0.986292 0.961973
0.998846062658 0.914022620358 0.88593495935
0.987149 0.963515
0.999487705841 0.921002586538 0.88718699187
0.986292 0.960432
0.999662372039 0.933134557944 0.89454200542
0.992118 0.96557
0.999476301518 0.925180030448 0.900401084011
0.992461 0.964029
0.999587343603 0.917383539923 0.89535501355
0.992461 0.967626
0.998926493141 0.918790068957 0.903588075881
0.991947 0.967112
0.999671375451 0.920070168415 0.891387533875
0.991432 0.96814
0.999658170447 0.917315057236 0.892552845528
0.994688 0.970195
0.999907264853 0.925986018996 0.891176151762
0.996059 0.969168
0.999950481232 0.927376744333 0.895642276423
0.995374 0.971223
0.99982923528 0.921413482661 0.890249322493
0.995888 0.96814
0.999794422086 0.920286152274 0.903116531165
0.996059 0.970709
0.999876653252 0.927208171565 0.905100271003
0.997601 0.970709
0.999894059849 0.927192367868 0.904292682927
0.99623 0.971223
0.999887457346 0.911109472209 0.906184281843
0.996059 0.972765
0.999723595243 0.916293084829 0.908856368564
0.996573 0.970195
0.999859246654 0.918358101238 0.906601626016
0.995031 0.967626
0.999784218219 0.931111684727 0.905956639566
0.997258 0.969168
0.999924071223 0.929115151004 0.915875338753
0.998115 0.970195
0.998854465843 0.934488407988 0.90172899729
0.997258 0.970709
0.999944478958 0.929731495188 0.893376693767
0.995374 0.964543
0.999940877593 0.932663080983 0.897447154472
0.998801 0.966084
0.999973890104 0.918005152005 0.883761517615
0.839616 0.83556
0.90956282431 0.866016256736 0.834975609756
0.876114 0.868448
0.945359191133 0.881656648879 0.858189701897
0.942084 0.929085
0.972817797945 0.904319150393 0.878894308943
0.958705 0.939363
0.986478375304 0.908054090787 0.89660704607
0.95682 0.935766
0.98950952411 0.910450984834 0.890634146341
0.961618 0.931141
0.994918173988 0.90936052974 0.893577235772
0.97464 0.953751
0.997470941487 0.925928072107 0.911138211382
0.983722 0.956321
0.997559775155 0.916904161113 0.911262872629
0.983893 0.959404
0.998077771475 0.924410917194 0.911420054201
0.987491 0.956835
0.998841260838 0.91813684948 0.899051490515
0.987663 0.956321
0.998927093368 0.919648736494 0.913555555556
0.984921 0.959918
0.998938497691 0.926734060655 0.906612466125
0.986121 0.959404
0.999490706978 0.918394976532 0.909864498645
0.989205 0.961973
0.999496109025 0.918173724773 0.905615176152
0.991947 0.961459
0.999541126087 0.911393938755 0.90220596206
0.991261 0.965057
0.999653368627 0.918321225945 0.904785907859
0.991432 0.96557
0.999544727452 0.915160486543 0.898829268293
0.992803 0.964543
0.999567536096 0.919042928109 0.892444444444
0.99366 0.964029
0.999729597517 0.921935004662 0.90687804878
0.994345 0.966084
0.999868850294 0.921692681308 0.910037940379
0.99743 0.970195
0.999895860531 0.916743490194 0.906010840108
0.997087 0.970709
0.999856845745 0.912315821081 0.894623306233
0.991432 0.963001
0.999811828683 0.923994753173 0.906130081301
0.991604 0.961973
0.999823233005 0.922714653715 0.917067750678
0.992975 0.960946
0.999925871905 0.930284624583 0.91993495935
0.996744 0.963001
0.999930673725 0.922135184824 0.911111111111
0.995031 0.966598
0.999695384551 0.920486332436 0.899414634146
0.995031 0.963515
0.999766211394 0.929173097893 0.902352303523
0.996573 0.967626
0.999819031413 0.927864024991 0.904016260163
0.997772 0.969168
0.999914467583 0.92347849907 0.903344173442
0.994688 0.964029
0.999867649839 0.926159859663 0.906135501355
0.998286 0.970195
0.999894059849 0.923836716202 0.916292682927
[[0.93852888652418753, 0.89435772357723575], [0.94038318697353929, 0.89081842818428181], [0.94263784774718296, 0.90014092140921409]]
0.898704607046
size of different sets: 10953 1946 1945
0.784441 0.761562
0.915142839136 0.867886360883 0.859550135501
0.898389 0.881809
0.942776112147 0.88598686186 0.877073170732
0.922207 0.903392
0.974343876329 0.900057420099 0.903208672087
SAVED

0.915867 0.893114
0.985365853659 0.900036348503 0.901035230352
SAVED

0.938143 0.9111
0.987775166788 0.888610275564 0.891338753388
SAVED

0.956991 0.92446
0.993544553386 0.892071285209 0.910016260163
SAVED

0.973269 0.945015
0.993267248287 0.892561199817 0.900531165312
SAVED

0.972927 0.944502
0.997876995381 0.902917889258 0.90833604336
SAVED

0.982865 0.959404
0.998477222867 0.908644095475 0.898048780488
SAVED

0.986977 0.955807
0.99835477646 0.90084760495 0.903105691057
SAVED

0.98989 0.963515
0.998825955037 0.896717572131 0.900395663957
0.988691 0.961973
0.998866170279 0.89471050261 0.913376693767
0.984578 0.947071
0.999081651946 0.888826259423 0.907300813008
0.990747 0.959918
0.999532422788 0.891396994137 0.898463414634
0.994688 0.964543
0.99965606965 0.898745713247 0.905371273713
0.993146 0.963515
0.99912126696 0.882794515064 0.904845528455
0.995888 0.965057
0.999637462598 0.888373220109 0.902097560976
0.995031 0.96557
0.999274925197 0.905788894215 0.914059620596
SAVED

0.991604 0.95889
0.999539625518 0.899546433896 0.921382113821
0.99486 0.965057
0.99967047511 0.904429776272 0.909387533875
SAVED

0.994517 0.96814
0.999501811186 0.896822930111 0.900617886179
0.992118 0.961973
0.999651267831 0.896717572131 0.899452574526
0.995031 0.961459
0.999698385688 0.89761838286 0.891891598916
0.995716 0.964543
0.999772513783 0.888273130028 0.891674796748
0.99486 0.966598
0.999717292854 0.891112527591 0.879962059621
0.997772 0.967626
0.999200496988 0.884517118038 0.8832899729
0.99623 0.965057
0.999809727887 0.891265296662 0.894639566396
0.996402 0.96557
0.99958464258 0.885491679354 0.891262872629
0.995888 0.967112
0.999792921517 0.886782314609 0.894899728997
0.995716 0.966598
0.999824133347 0.884474974846 0.896260162602
0.994517 0.963001
0.99988535655 0.88519140911 0.88674796748
0.997087 0.966598
0.999983793858 0.88111932318 0.890422764228
0.998629 0.971223
0.999989796133 0.886392490083 0.892200542005
0.998458 0.970709
0.99998559454 0.884917478362 0.886059620596
0.897875 0.881295
0.925229061814 0.85944191878 0.860943089431
0.888451 0.862282
0.947985486499 0.876794378098 0.876677506775
0.945168 0.919322
0.978985735594 0.891044044904 0.89979403794
0.958191 0.922405
0.989330956432 0.897528828577 0.897116531165
0.973269 0.941418
0.993345878088 0.894626216226 0.901566395664
0.979609 0.948613
0.996527683992 0.906842474016 0.898124661247
SAVED

0.982865 0.952724
0.997545069581 0.895289971501 0.893929539295
0.972584 0.932169
0.998310959854 0.884638279715 0.892872628726
0.980637 0.94964
0.998508434697 0.889732338052 0.891078590786
0.988005 0.955293
0.998389589654 0.890591005589 0.891669376694
0.986977 0.954265
0.998834958449 0.890764846256 0.902991869919
0.984578 0.953751
0.998312760536 0.887193210732 0.90260704607
0.980637 0.947585
0.998361378963 0.893986166497 0.896211382114
0.991775 0.963515
0.99912606878 0.903871378978 0.898986449864
0.992803 0.964029
0.99955102984 0.88936885302 0.893528455285
0.996916 0.970195
0.999632060551 0.893040578626 0.889165311653
0.994688 0.971737
0.996195157965 0.892537494271 0.893604336043
0.991432 0.961973
0.999683079887 0.896433105584 0.894769647696
0.993831 0.96557
0.99965366874 0.895658724431 0.884227642276
0.998115 0.968654
0.999748504683 0.896585874656 0.882850948509
0.994688 0.963001
0.999375163187 0.887672589541 0.896005420054
0.995374 0.966084
0.999645865783 0.88276290767 0.889414634146
0.995031 0.967112
0.999639263281 0.88905277908 0.880281842818
0.996916 0.970195
0.999730497859 0.89518988142 0.886070460705
0.995374 0.968654
0.999729297404 0.890448772316 0.894368563686
0.99743 0.972251
0.999752106048 0.886455704871 0.893891598916
0.997601 0.969681
0.999547428475 0.8804871753 0.891864498645
0.995031 0.961973
0.9997707131 0.893045846525 0.894704607046
0.996744 0.967112
0.999716692627 0.894736842105 0.896886178862
0.99743 0.96814
0.99977311401 0.878322068809 0.891986449864
0.998115 0.971737
0.999872751773 0.887367051399 0.897008130081
0.997944 0.971223
0.999862547906 0.885528554647 0.897111111111
0.998115 0.971737
0.999867949953 0.885665520021 0.896655826558
0.998801 0.972765
0.999884156095 0.885834092789 0.893306233062
0.833619 0.828366
0.911223953879 0.86049549858 0.840140921409
0.920493 0.898767
0.95918873253 0.885575965738 0.865317073171
0.947224 0.914697
0.9797567278 0.894178444811 0.888352303523
0.953393 0.92703
0.986847215095 0.903987272756 0.890823848238
0.964873 0.942446
0.993657996381 0.90507772785 0.894184281843
0.965559 0.944502
0.994213206805 0.900837069152 0.898948509485
0.984064 0.963001
0.998490427872 0.911989211343 0.90139295393
SAVED

0.978239 0.954265
0.99893279553 0.913516902054 0.90989701897
SAVED

0.988177 0.963001
0.998972410544 0.905862644801 0.89627100271
0.98852 0.963001
0.999378164324 0.905225229022 0.914249322493
0.990918 0.966598
0.999613453499 0.916066565172 0.90566395664
SAVED

0.989719 0.961973
0.999227507225 0.919485431625 0.898970189702
SAVED

0.987834 0.957862
0.998746124781 0.923046531352 0.91233604336
SAVED

0.992803 0.964029
0.997863190149 0.925870125218 0.908178861789
SAVED

0.993146 0.966084
0.999547428475 0.926465397805 0.903414634146
SAVED

0.994345 0.968654
0.999783317877 0.919464360029 0.903853658537
0.994517 0.968654
0.999783317877 0.908148912969 0.910807588076
0.99486 0.966598
0.999647666466 0.913390472478 0.905653116531
0.995031 0.961459
0.999726896494 0.912726717203 0.899211382114
0.997944 0.971737
0.999914767697 0.911736352191 0.901972899729
0.998115 0.968654
0.999918369062 0.913674939024 0.902308943089
0.99623 0.962487
0.999919569517 0.914338694299 0.902997289973
0.998286 0.970709
0.999796522882 0.905320051204 0.907403794038
0.993831 0.957862
0.999867349726 0.907300781229 0.905376693767
0.997258 0.96814
0.999921970427 0.909587049397 0.896829268293
0.997087 0.966598
0.999974190218 0.907859178524 0.906097560976
0.99623 0.963001
0.999917768834 0.913495830458 0.89937398374
0.995888 0.962487
0.999893759735 0.920865621164 0.915379403794
0.998629 0.96814
0.999914167469 0.921634734419 0.914907859079
0.995888 0.964543
0.999709189783 0.919042928109 0.913566395664
0.998629 0.970709
0.999801024588 0.921837548531 0.911436314363
0.998801 0.970195
0.999829535394 0.921729556601 0.909764227642
[[0.92304653135190096, 0.9123360433604335], [0.92587012521795931, 0.90817886178861784], [0.92646539780539328, 0.90341463414634149]]
0.919181571816
size of different sets: 11022 1945 1945
0.840158 0.83856
0.914895451798 0.865026210097 0.84091598916
0.887613 0.880206
0.955761708332 0.901271654793 0.870742547425
0.934555 0.912596
0.975942356246 0.892905216849 0.861951219512
SAVED

0.950146 0.930591
0.988756623164 0.908467119735 0.863024390244
SAVED

0.971218 0.947558
0.992804606285 0.890317038204 0.87091598916
0.979955 0.953213
0.996876296481 0.898437243508 0.873463414634
SAVED

0.984581 0.956812
0.997840581772 0.889977784344 0.883842818428
0.9767 0.94653
0.998289238173 0.886464865338 0.873490514905
0.977728 0.953213
0.998142846983 0.900067850772 0.894032520325
SAVED

0.984239 0.958869
0.999035418371 0.894918852665 0.878254742547
0.989721 0.962982
0.99886235672 0.906655941867 0.881723577236
SAVED

0.991263 0.968123
0.999028306248 0.918863609004 0.901555555556
SAVED

0.990063 0.960411
0.999262413617 0.915569563457 0.897571815718
SAVED

0.993147 0.967095
0.999385690409 0.89488054981 0.875230352304
0.995203 0.96401
0.999462145727 0.903657375488 0.88139295393
0.995032 0.967095
0.99932464469 0.91958041958 0.890075880759
SAVED

0.995203 0.969666
0.999738333156 0.912893835429 0.900921409214
0.995546 0.973779
0.999770930384 0.915717303041 0.897593495935
SAVED

0.996574 0.973779
0.999736555125 0.90482834849 0.891495934959
0.992633 0.969152
0.9996423195 0.916483360145 0.887317073171
SAVED

0.995203 0.968123
0.999606758887 0.904292108517 0.892054200542
0.996402 0.97018
0.999751964724 0.904883066855 0.902368563686
0.995203 0.967095
0.999660692484 0.916094859757 0.902466124661
0.995717 0.968123
0.999728850325 0.895673966097 0.894655826558
0.996745 0.973779
0.999749594016 0.90422644648 0.891420054201
0.995203 0.969152
0.999720552849 0.907339921424 0.900997289973
0.996402 0.970694
0.999692104359 0.902924149403 0.893192411924
0.996231 0.968123
0.999743667247 0.899958414043 0.890910569106
0.99743 0.972237
0.999735962448 0.906902174508 0.891116531165
0.998801 0.973779
0.999774486445 0.906841984307 0.890211382114
0.997773 0.972751
0.999829605396 0.909714698447 0.900173441734
0.998287 0.972237
0.999793452106 0.898010440264 0.885192411924
0.997773 0.970694
0.999807676351 0.90255753636 0.891447154472
0.998458 0.973265
0.999789896044 0.89426223229 0.882894308943
0.84958 0.840103
0.90985769828 0.864479026451 0.839143631436
0.901491 0.881748
0.941489456278 0.871368068551 0.828
0.919822 0.900257
0.977209499425 0.882782319402 0.864509485095
0.942265 0.931105
0.98977899079 0.919509285706 0.892655826558
SAVED

0.955971 0.927506
0.99286446665 0.879285815906 0.884325203252
0.969162 0.947558
0.994035596174 0.895383958764 0.888086720867
0.974131 0.94653
0.997225383166 0.890825918995 0.885344173442
0.963337 0.939332
0.997124628095 0.896385304836 0.897159891599
0.976872 0.946015
0.998845169091 0.902459043304 0.895550135501
0.985609 0.959897
0.998239453315 0.89339768213 0.89620596206
0.990063 0.959897
0.998804867063 0.904264749335 0.897051490515
0.983725 0.954756
0.99891569764 0.908100506692 0.894054200542
0.985609 0.955784
0.998560684186 0.895805290171 0.903474254743
0.989721 0.961954
0.999381541671 0.907947295271 0.907826558266
0.99349 0.964524
0.999469850527 0.903394727338 0.901799457995
0.994175 0.966581
0.999485260126 0.899044617354 0.902986449864
0.99606 0.970694
0.999634022024 0.894032415159 0.901571815718
0.990749 0.96144
0.999636985408 0.895520754676 0.888140921409
0.996231 0.967095
0.999610907625 0.878005406174 0.888623306233
0.995717 0.968123
0.999506003817 0.888122831785 0.888845528455
0.989721 0.957326
0.998804570724 0.896543988093 0.896059620596
0.990749 0.960411
0.999266562356 0.87567987568 0.893338753388
0.991948 0.965553
0.999443180067 0.880396598706 0.890655826558
0.99606 0.968638
0.99954541683 0.876845376845 0.886086720867
0.99606 0.973265
0.99955104726 0.875461002222 0.887327913279
0.990577 0.96144
0.999437253298 0.867636276087 0.875414634146
0.992805 0.965553
0.999440216683 0.891307440603 0.881588075881
0.996745 0.971722
0.999684399559 0.881649649255 0.886422764228
0.998287 0.971208
0.999666619253 0.884380095648 0.887279132791
0.997602 0.971722
0.999692104359 0.88378913731 0.878303523035
0.99743 0.969666
0.99969566042 0.889124177857 0.88156097561
0.997773 0.971208
0.999901911976 0.882366459831 0.881398373984
0.998115 0.970694
0.999924433697 0.88300119286 0.879170731707
0.998287 0.971722
0.999882353638 0.88545804743 0.879875338753
0.854206 0.848329
0.910523274421 0.872506210534 0.845945799458
0.906973 0.88946
0.951086673068 0.872429604824 0.838937669377
0.934213 0.908483
0.972783981129 0.876478763803 0.872807588076
0.93901 0.922879
0.983874743667 0.892724646246 0.869615176152
0.947576 0.924422
0.99049612982 0.883898574039 0.869046070461
0.931129 0.901799
0.994749179143 0.893102202961 0.86325203252
0.973788 0.951671
0.994971729313 0.906661413704 0.883349593496
0.979099 0.952185
0.997166708154 0.881036803572 0.875387533875
0.983553 0.957841
0.998512677359 0.884568874005 0.873176151762
0.985438 0.957326
0.998113805815 0.88868643094 0.872124661247
0.981326 0.956812
0.998485414222 0.894070718014 0.888059620596
0.986637 0.961954
0.999345388381 0.903586241614 0.88725203252
0.992633 0.967095
0.99962335384 0.893857316393 0.875279132791
0.993147 0.97018
0.999677880113 0.894130908215 0.878216802168
0.994518 0.969666
0.999306271707 0.902185451481 0.883414634146
0.991434 0.964524
0.999262413617 0.900636921764 0.893013550136
0.991948 0.966067
0.999420065669 0.906792737779 0.890926829268
0.995374 0.967095
0.999338276259 0.886196745352 0.874693766938
0.995032 0.965553
0.999510745232 0.89489696532 0.869761517615
0.996745 0.969152
0.999699216481 0.895542642022 0.868227642276
0.99486 0.968638
0.999652395007 0.894814887773 0.866330623306
0.994346 0.969666
0.999692697035 0.881233789684 0.857398373984
0.993147 0.960925
0.999648246269 0.885622202524 0.867750677507
0.997088 0.972237
0.999799378875 0.878607308185 0.870796747967
0.995717 0.969666
0.999682621528 0.882278910448 0.870634146341
0.991948 0.964524
0.99973181371 0.891597447935 0.857840108401
0.994004 0.967609
0.999814788474 0.899279906322 0.872108401084
0.99606 0.97018
0.999821307919 0.887406021209 0.87664498645
0.993661 0.969666
0.999800564228 0.903558882432 0.886151761518
0.995374 0.971208
0.999853312471 0.887088654694 0.88345799458
0.995717 0.972237
0.998704111992 0.895619247732 0.889273712737
0.99606 0.970694
0.999914950867 0.891772546702 0.879913279133
[[0.91886360900445407, 0.90155555555555555], [0.91950928570646873, 0.8926558265582657], [0.91958041958041958, 0.89007588075880761]]
0.914780487805
size of different sets: 11022 1945 1945
0.86637 0.861183
0.936528158079 0.867491272421 0.857506775068
0.884358 0.876093
0.960557649681 0.88034461626 0.881962059621
0.925133 0.908997
0.981860827614 0.89712126684 0.891116531165
SAVED

0.951002 0.933676
0.989898118844 0.89779977456 0.89962601626
SAVED

0.95717 0.937275
0.993889501322 0.89083412675 0.893495934959
SAVED

0.965051 0.94653
0.99402670602 0.8937834466 0.896308943089
SAVED

0.97927 0.955784
0.997575062527 0.894161003316 0.912346883469
SAVED

0.979613 0.960925
0.998826796107 0.897334668461 0.908563685637
SAVED

0.972589 0.952185
0.996908301032 0.88646212942 0.892417344173
0.979955 0.960925
0.997802354113 0.890352605141 0.896840108401
0.985095 0.960925
0.998251899529 0.895632927323 0.902184281843
0.986294 0.96401
0.998781752664 0.895890103637 0.895962059621
0.979784 0.950643
0.998935255977 0.889750703131 0.899707317073
0.991091 0.965039
0.999351907827 0.894631581251 0.901311653117
0.989378 0.966581
0.999429548499 0.895939350165 0.907127371274
0.993318 0.968123
0.999559937413 0.889389561925 0.899224932249
0.991777 0.969152
0.99928671337 0.903539731005 0.90454200542
SAVED

0.993661 0.972751
0.999267747709 0.891649430382 0.899002710027
0.99349 0.968638
0.999460960374 0.89990095976 0.896585365854
SAVED

0.993661 0.968638
0.999600832118 0.897909211289 0.897712737127
SAVED

0.996231 0.971208
0.999647060915 0.901969313941 0.902162601626
SAVED

0.992119 0.962468
0.999789303368 0.897345612134 0.905853658537
0.995203 0.968638
0.999426585114 0.89954529039 0.905766937669
0.997088 0.972237
0.999759669523 0.899003578581 0.901956639566
0.995717 0.972237
0.999890651115 0.898221105968 0.899566395664
0.994689 0.969666
0.999818344535 0.888787659914 0.899598915989
0.997088 0.974293
0.99972233088 0.909110060519 0.914482384824
SAVED

0.993147 0.963496
0.999372651518 0.895654814669 0.914113821138
0.996916 0.972237
0.99957001292 0.902237433928 0.909907859079
SAVED

0.996916 0.975321
0.999766188969 0.904746270943 0.910075880759
SAVED

0.997088 0.974293
0.999788710691 0.900344178513 0.908780487805
0.996916 0.974807
0.999754335431 0.899512459372 0.908758807588
0.996916 0.974293
0.999792859429 0.911320682447 0.912287262873
SAVED

0.996231 0.974293
0.99979641549 0.912300141173 0.922005420054
SAVED

0.815316 0.803085
0.922853324325 0.863764951793 0.845452574526
0.886928 0.879692
0.956983215391 0.885707015989 0.865804878049
0.908001 0.885347
0.97221501132 0.876766035217 0.87081300813
0.943121 0.926992
0.985142479523 0.889198047649 0.899127371274
0.940894 0.928021
0.990618813935 0.887923109754 0.891262872629
0.935583 0.911568
0.993810378958 0.898073366383 0.899799457995
0.971561 0.948586
0.996214869078 0.890155619029 0.897089430894
0.969162 0.947558
0.998350283892 0.895523490594 0.892720867209
0.970875 0.9491
0.997643220369 0.891118662245 0.889756097561
0.979613 0.957841
0.998428517241 0.897088435821 0.881550135501
0.975158 0.949614
0.998564832924 0.896437287282 0.903035230352
0.985266 0.962468
0.998880729704 0.89614727995 0.886758807588
0.990749 0.968123
0.998996894373 0.896951639909 0.898590785908
0.990235 0.966581
0.999184180269 0.902543856769 0.887669376694
0.993318 0.966067
0.999519635385 0.891217155302 0.88779403794
0.994004 0.968123
0.999427177791 0.889608435383 0.894195121951
0.99349 0.968638
0.99950541114 0.882976569596 0.89498102981
0.99092 0.965039
0.998948887545 0.892437374832 0.892460704607
0.990749 0.964524
0.999041937816 0.889411449271 0.895723577236
0.992462 0.969152
0.999631651316 0.898576775337 0.899750677507
0.994689 0.967609
0.999487630833 0.902768202064 0.895734417344
0.992462 0.965039
0.999409397485 0.906385085963 0.898471544715
0.99606 0.971208
0.999573568982 0.896021427712 0.89827100271
0.992976 0.966067
0.999436067944 0.901438545805 0.890075880759
0.992119 0.969152
0.999367317426 0.913367149283 0.897317073171
SAVED

0.997088 0.970694
0.999382134348 0.893263622137 0.879739837398
0.99743 0.972237
0.999244040634 0.892803987874 0.888758807588
0.998629 0.972237
0.999208480021 0.899884544251 0.896726287263
0.998629 0.973265
0.99945977502 0.891966796896 0.892471544715
0.996574 0.97018
0.999652987684 0.896185582805 0.894292682927
0.998458 0.974807
0.999767967 0.888519539928 0.894081300813
0.998287 0.972751
0.999814195797 0.900355122186 0.902520325203
0.998287 0.973265
0.999811825089 0.899052825109 0.898850948509
0.998458 0.973265
0.99980026789 0.898708099412 0.898070460705
0.869111 0.85347
0.923707964392 0.856657036235 0.834043360434
0.920678 0.897686
0.969971136636 0.878949297963 0.878298102981
0.910228 0.884319
0.981167988336 0.883676964663 0.881544715447
0.951688 0.927506
0.987429026943 0.87223535463 0.885859078591
0.949803 0.932134
0.994526925311 0.89605425873 0.893105691057
0.965564 0.948072
0.995385121439 0.898636965538 0.890596205962
0.973445 0.951157
0.99673938812 0.895922934655 0.88887804878
0.979442 0.956298
0.998583205908 0.90767096753 0.894113821138
0.979442 0.950643
0.998474746038 0.887118749795 0.882195121951
0.989035 0.966067
0.999004599173 0.886642700023 0.896086720867
0.984924 0.956812
0.999241669926 0.898144500257 0.89572899729
0.988864 0.962982
0.999514301293 0.896234829333 0.891127371274
0.988693 0.96401
0.999487630833 0.894593278396 0.898
0.993832 0.972751
0.999646468238 0.898412620244 0.897398373984
0.99486 0.975321
0.999518450031 0.887616686912 0.885149051491
0.994689 0.970694
0.9996423195 0.892617945435 0.894504065041
0.996574 0.973779
0.999900133945 0.886670059205 0.898449864499
0.995203 0.969666
0.999902504653 0.896284075862 0.897739837398
0.994689 0.969666
0.999946955419 0.893411361721 0.900715447154
0.993661 0.968123
0.999801156905 0.886347220854 0.894997289973
0.995888 0.971208
0.999799971552 0.891020169189 0.897029810298
0.99486 0.967609
0.999787525337 0.88925823785 0.908195121951
0.995374 0.972751
0.999780413214 0.88070028563 0.900433604336
0.997259 0.975321
0.99980412029 0.899227923876 0.913853658537
0.995032 0.974293
0.999751372047 0.898658852884 0.914027100271
0.997944 0.977378
0.999846200348 0.896415399937 0.910211382114
0.997259 0.975836
0.999565271505 0.89265624829 0.907279132791
0.997088 0.974293
0.999581866458 0.888853321952 0.908346883469
0.995546 0.972237
0.999690919005 0.883129781017 0.901360433604
0.991434 0.960411
0.999654765715 0.888754828896 0.899338753388
0.997944 0.974293
0.999905468037 0.889542773346 0.906921409214
0.99743 0.971208
0.999919692282 0.901247031529 0.910585365854
[[0.91132068244744291, 0.91228726287262873], [0.91230014117338065, 0.9220054200542005], [0.91336714928264229, 0.89731707317073184]]
0.920964769648
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869062
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.683345 0.673176
0.863677833899 0.83262831285 0.811040650407
0.820082 0.812436
0.884973304883 0.851750786234 0.820054200542
0.777759 0.76927
0.899042637159 0.85267266856 0.835886178862
SAVED

0.792324 0.7852
0.929812699013 0.870673079456 0.85989701897
SAVED

0.818197 0.804214
0.940985933669 0.876984022462 0.859566395664
SAVED

0.8756 0.861768
0.944809382756 0.877721528323 0.850569105691
SAVED

0.91501 0.912127
0.956774917694 0.892461109736 0.85898102981
SAVED

0.889308 0.880267
0.9637957786 0.884975425251 0.865707317073
SAVED

0.893934 0.877184
0.972324410952 0.897813295124 0.870590785908
SAVED

0.897875 0.880781
0.978806867803 0.898556068883 0.878455284553
SAVED

0.935744 0.925488
0.980890257408 0.895990602068 0.866352303523
SAVED

0.897533 0.877698
0.984680693983 0.900015276907 0.864471544715
SAVED

0.914667 0.894656
0.987891310807 0.908480790606 0.888558265583
SAVED

0.920493 0.902364
0.992672122735 0.919348466251 0.888379403794
SAVED

0.94414 0.922405
0.994976996282 0.922835815392 0.886785907859
SAVED

0.955963 0.923433
0.995874336373 0.916856750022 0.892531165312
SAVED


Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py", line 297, in <module>
    dropout : 0.1,learning_rate : lr}) #sgd
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1306, in _run_fn
    status, run_metadata)
KeyboardInterrupt
>>> 
KeyboardInterrupt
>>> layer4_DO[1::2]
<tf.Tensor 'strided_slice:0' shape=(?, 1, ?, 512) dtype=float32>
>>> layer4_DO[1:::2]
SyntaxError: invalid syntax
>>> layer4_DO[:,:,:,1::2]
<tf.Tensor 'strided_slice_1:0' shape=(?, 1, ?, 256) dtype=float32>
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
737990
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.654387 0.650051
0.857843622733 0.840203551618 0.818265582656
0.797635 0.792395
0.892866896554 0.868824046905 0.842796747967
0.796607 0.78777
0.904739096117 0.871479068003 0.853588075881
SAVED

0.770562 0.761562
0.916246357369 0.870483435092 0.855669376694
SAVED

0.813057 0.794964
0.935651711999 0.876425625168 0.876504065041
SAVED

0.855894 0.848921
0.941173804872 0.870841652224 0.858964769648
SAVED

0.898903 0.886434
0.951356063948 0.891797354461 0.87247696477
SAVED


 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}

 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer5_avg_pooling.py SAVED
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869062
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.701337 0.692189
0.862602826471 0.840356320689 0.816829268293
0.792666 0.796506
0.893946705801 0.85637073366 0.832991869919
0.788897 0.788798
0.912589171296 0.873886497848 0.850552845528
SAVED

0.75891 0.75591
0.927792333294 0.888215183139 0.860894308943
SAVED

0.800377 0.805755
0.942008121078 0.890346048286 0.867197831978
SAVED

0.884681 0.871531
0.948959955823 0.883136928499 0.862384823848
SAVED

0.880398 0.875642
0.957404556327 0.907674802059 0.869636856369
SAVED

0.880569 0.865879
0.964369596077 0.904924958779 0.876130081301
SAVED

0.904558 0.885406
0.972363425738 0.902480653641 0.870260162602
SAVED

