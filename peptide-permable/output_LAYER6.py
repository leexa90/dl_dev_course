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

0.880398 0.874101
0.978393311065 0.909476423518 0.884601626016
SAVED

0.928718 0.909558
0.977838400754 0.90697943939 0.874233062331
SAVED

0.913639 0.887975
0.985483198132 0.903233963198 0.881176151762
0.937971 0.910072
0.989085763504 0.907843374827 0.877739837398
SAVED

0.914325 0.883864
0.99308387879 0.913458955165 0.879257452575
SAVED

0.954078 0.924974
0.994223710786 0.908854811436 0.886910569106
SAVED

0.956306 0.924974
0.995101243371 0.910024285014 0.875512195122
SAVED

0.944654 0.917266
0.995829319312 0.902217258691 0.893490514905
0.953564 0.925488
0.996679541546 0.912157784111 0.89660704607
SAVED

0.966073 0.937307
0.998172607418 0.9116889411 0.881739837398
SAVED

0.966073 0.933196
0.998475722299 0.920897228558 0.890363143631
SAVED

0.961618 0.926516
0.998173207646 0.919611861201 0.896341463415
SAVED

0.958191 0.920349
0.998775836042 0.917367736226 0.898162601626
SAVED

0.972755 0.944502
0.998791441957 0.908191056161 0.894850948509
0.967272 0.939363
0.998850864478 0.910782862471 0.885409214092
0.978753 0.946557
0.999207999832 0.903908254271 0.891609756098
0.974126 0.941932
0.999197195737 0.909803033256 0.889712737127
0.975326 0.94964
0.999157580723 0.910540539117 0.896086720867
0.969671 0.934224
0.999222405292 0.89730230892 0.888059620596

 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
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
0.765764 0.751285
0.910371930962 0.855733317881 0.833317073171
0.910384 0.906475
0.953621322481 0.880571461684 0.855279132791

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 301, in <module>
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
>>> size of different sets: 10953 1946 1945data_xgb = pd.read_csv('XGB_all.csv')
KeyboardInterrupt
>>> data_xgb = pd.read_csv('XGB_all.csv')
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> 
KeyboardInterrupt
>>> data_xgb.keys()[-5:-2]
Index([u'Gwif', u'Goct', u'y'], dtype='object')
>>> data_xgb.keys()[-5:-1]
Index([u'Gwif', u'Goct', u'y', u'testPred'], dtype='object')
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[data_xgb.keys()[-5:-1]],on='seq')

Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    data.merge(data_xgb[data_xgb.keys()[-5:-1]],on='seq')
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'seq'
>>> 
KeyboardInterrupt
>>> data.iloc[0]
seq                                                     SPCG
source                                                    -1
size                                                       4
type_aa                                                    4
len                                                        4
num_A                                                      0
per_A                                                      0
num_C                                                      1
per_C                                                   0.25
num_E                                                      0
per_E                                                      0
num_D                                                      0
per_D                                                      0
num_G                                                      1
per_G                                                   0.25
num_F                                                      0
per_F                                                      0
num_I                                                      0
per_I                                                      0
num_H                                                      0
per_H                                                      0
num_K                                                      0
per_K                                                      0
num_M                                                      0
per_M                                                      0
num_L                                                      0
per_L                                                      0
num_N                                                      0
per_N                                                      0
num_Q                                                      0
per_Q                                                      0
num_P                                                      1
per_P                                                   0.25
num_S                                                      1
per_S                                                   0.25
num_R                                                      0
per_R                                                      0
num_T                                                      0
per_T                                                      0
num_W                                                      0
per_W                                                      0
num_V                                                      0
per_V                                                      0
num_Y                                                      0
per_Y                                                      0
X          [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...
Name: 0, dtype: object
>>> 
KeyboardInterrupt
>>> data
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E  \
0     0.000000    1.0  0.250000    0.0   
1     0.000000    0.0  0.000000    0.0   
2     0.250000    0.0  0.000000    0.0   
3     0.250000    0.0  0.000000    0.0   
4     0.250000    0.0  0.000000    0.0   
5     0.000000    0.0  0.000000    0.0   
6     0.000000    0.0  0.000000    0.0   
7     0.250000    0.0  0.000000    0.0   
8     0.000000    0.0  0.000000    0.0   
9     0.000000    0.0  0.000000    1.0   
10    0.000000    0.0  0.000000    0.0   
11    0.000000    0.0  0.000000    0.0   
12    0.000000    0.0  0.000000    0.0   
13    0.000000    0.0  0.000000    0.0   
14    0.000000    0.0  0.000000    0.0   
15    0.000000    0.0  0.000000    0.0   
16    0.000000    0.0  0.000000    0.0   
17    0.000000    0.0  0.000000    0.0   
18    0.000000    0.0  0.000000    0.0   
19    0.000000    0.0  0.000000    1.0   
20    0.000000    0.0  0.000000    1.0   
21    0.000000    0.0  0.000000    0.0   
22    0.000000    0.0  0.000000    0.0   
23    0.000000    0.0  0.000000    0.0   
24    0.250000    0.0  0.000000    1.0   
25    0.250000    0.0  0.000000    0.0   
26    0.250000    0.0  0.000000    0.0   
27    0.250000    0.0  0.000000    0.0   
28    0.250000    0.0  0.000000    0.0   
29    0.250000    0.0  0.000000    0.0   
...        ...    ...       ...    ...   
9697  0.300000    0.0  0.000000    0.0   
9698  0.033333    9.0  0.300000    2.0   
9699  0.033333    0.0  0.000000    3.0   
9700  0.066667    2.0  0.066667    3.0   
9701  0.066667    0.0  0.000000    1.0   
9702  0.033333    2.0  0.066667    0.0   
9703  0.033333   11.0  0.366667    0.0   
9704  0.066667    6.0  0.200000    0.0   
9705  0.000000    6.0  0.200000    1.0   
9706  0.133333    0.0  0.000000    5.0   
9707  0.166667    0.0  0.000000    3.0   
9708  0.100000    0.0  0.000000    2.0   
9709  0.000000    0.0  0.000000    2.0   
9710  0.100000    0.0  0.000000    8.0   
9711  0.066667    4.0  0.133333    0.0   
9712  0.200000    0.0  0.000000    0.0   
9713  0.066667    0.0  0.000000    3.0   
9714  0.100000    0.0  0.000000    1.0   
9715  0.166667    0.0  0.000000    3.0   
9716  0.000000    6.0  0.200000    2.0   
9717  0.100000    0.0  0.000000    0.0   
9718  0.000000    2.0  0.066667    1.0   
9719  0.066667    0.0  0.000000    0.0   
9720  0.066667    0.0  0.000000    0.0   
9721  0.000000    1.0  0.033333    0.0   
9722  0.033333    1.0  0.033333    0.0   
9723  0.066667    0.0  0.000000    0.0   
9724  0.466667    0.0  0.000000    7.0   
9725  0.400000    1.0  0.033333    2.0   
9726  0.400000    1.0  0.033333    2.0   

                            ...                             per_R  num_T  \
0                           ...                          0.000000    0.0   
1                           ...                          0.000000    0.0   
2                           ...                          0.000000    0.0   
3                           ...                          0.000000    1.0   
4                           ...                          0.000000    0.0   
5                           ...                          0.250000    0.0   
6                           ...                          0.250000    0.0   
7                           ...                          0.000000    0.0   
8                           ...                          0.000000    0.0   
9                           ...                          0.000000    0.0   
10                          ...                          0.000000    1.0   
11                          ...                          0.000000    0.0   
12                          ...                          0.000000    0.0   
13                          ...                          0.000000    1.0   
14                          ...                          0.000000    0.0   
15                          ...                          0.250000    0.0   
16                          ...                          0.250000    0.0   
17                          ...                          0.250000    0.0   
18                          ...                          0.000000    0.0   
19                          ...                          0.000000    0.0   
20                          ...                          0.000000    0.0   
21                          ...                          0.000000    0.0   
22                          ...                          0.250000    0.0   
23                          ...                          0.250000    0.0   
24                          ...                          0.000000    1.0   
25                          ...                          0.000000    1.0   
26                          ...                          0.000000    0.0   
27                          ...                          0.250000    0.0   
28                          ...                          0.000000    0.0   
29                          ...                          0.000000    0.0   
...                         ...                               ...    ...   
9697                        ...                          0.000000    2.0   
9698                        ...                          0.000000    3.0   
9699                        ...                          0.033333    1.0   
9700                        ...                          0.133333    0.0   
9701                        ...                          0.000000    2.0   
9702                        ...                          0.033333    2.0   
9703                        ...                          0.000000    3.0   
9704                        ...                          0.033333    1.0   
9705                        ...                          0.100000    0.0   
9706                        ...                          0.166667    3.0   
9707                        ...                          0.033333    1.0   
9708                        ...                          0.066667    2.0   
9709                        ...                          0.066667    2.0   
9710                        ...                          0.100000    1.0   
9711                        ...                          0.000000    3.0   
9712                        ...                          0.033333    1.0   
9713                        ...                          0.033333    1.0   
9714                        ...                          0.000000    1.0   
9715                        ...                          0.133333    2.0   
9716                        ...                          0.066667    1.0   
9717                        ...                          0.066667    3.0   
9718                        ...                          0.200000    1.0   
9719                        ...                          0.033333    2.0   
9720                        ...                          0.133333    0.0   
9721                        ...                          0.300000    0.0   
9722                        ...                          0.033333    0.0   
9723                        ...                          0.133333    0.0   
9724                        ...                          0.000000    0.0   
9725                        ...                          0.000000    0.0   
9726                        ...                          0.233333    0.0   

         per_T  num_W     per_W  num_V     per_V  num_Y     per_Y  \
0     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
2     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
4     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
7     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
8     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
9     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
10    0.250000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
11    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
13    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
14    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
16    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
20    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
22    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
23    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
24    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
25    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
26    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
27    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
28    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
29    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
...        ...    ...       ...    ...       ...    ...       ...   
9697  0.066667    1.0  0.033333    5.0  0.166667    2.0  0.066667   
9698  0.100000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9699  0.033333    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9700  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9701  0.066667    3.0  0.100000    1.0  0.033333    1.0  0.033333   
9702  0.066667    0.0  0.000000    0.0  0.000000    2.0  0.066667   
9703  0.100000    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9704  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9705  0.000000    0.0  0.000000    2.0  0.066667    1.0  0.033333   
9706  0.100000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9707  0.033333    1.0  0.033333    4.0  0.133333    0.0  0.000000   
9708  0.066667    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9709  0.066667    3.0  0.100000    3.0  0.100000    1.0  0.033333   
9710  0.033333    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9711  0.100000    1.0  0.033333    2.0  0.066667    2.0  0.066667   
9712  0.033333    1.0  0.033333    3.0  0.100000    0.0  0.000000   
9713  0.033333    0.0  0.000000    2.0  0.066667    0.0  0.000000   
9714  0.033333    1.0  0.033333    1.0  0.033333    2.0  0.066667   
9715  0.066667    0.0  0.000000    1.0  0.033333    4.0  0.133333   
9716  0.033333    4.0  0.133333    1.0  0.033333    1.0  0.033333   
9717  0.100000    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9718  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9719  0.066667    1.0  0.033333    1.0  0.033333    1.0  0.033333   
9720  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9721  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9722  0.000000    2.0  0.066667    4.0  0.133333    0.0  0.000000   
9723  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9724  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9725  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9726  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   

                                                      X  
0     [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...  
1     [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...  
2     [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....  
3     [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...  
4     [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...  
5     [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...  
6     [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...  
7     [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....  
8     [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...  
9     [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...  
10    [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...  
11    [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...  
12    [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...  
13    [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...  
14    [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...  
15    [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...  
16    [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...  
17    [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...  
18    [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...  
19    [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...  
20    [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...  
21    [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...  
22    [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...  
23    [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...  
24    [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...  
25    [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...  
26    [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....  
27    [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...  
28    [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...  
29    [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...  
...                                                 ...  
9697  [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...  
9698  [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...  
9699  [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...  
9700  [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....  
9701  [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...  
9702  [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...  
9703  [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...  
9704  [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...  
9705  [[14.0, 1.0, 14.0, 4.0, 15.0, 3.0, 14.0, 15.0,...  
9706  [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....  
9707  [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...  
9708  [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...  
9709  [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...  
9710  [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...  
9711  [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....  
9712  [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...  
9713  [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...  
9714  [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...  
9715  [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...  
9716  [[19.0, 1.0, 12.0, 8.0, 17.0, 9.0, 17.0, 16.0,...  
9717  [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...  
9718  [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....  
9719  [[4.0, 17.0, 16.0, 10.0, 11.0, 14.0, 0.0, 4.0,...  
9720  [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...  
9721  [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...  
9722  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...  
9723  [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...  
9724  [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...  
9725  [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...  
9726  [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...  

[9727 rows x 46 columns]
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[data_xgb.keys()[-5:-1]],on='seq')

Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    data.merge(data_xgb[data_xgb.keys()[-5:-1]],on='seq')
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'seq'
>>> data.merge(
KeyboardInterrupt
>>> data_xgb[data_xgb.keys()[-5:-1]]
       Gwif   Goct  y  testPred
0      1.38   1.73  0  0.007797
1      0.63   1.29  0  0.002599
2      4.24   7.08  0  0.031334
3      0.89   0.65  0  0.002652
4      0.92  -0.30  0  0.001054
5      0.16  -0.30  0  0.020269
6     -0.16  -1.73  0  0.039302
7      3.30   3.58  0  0.000825
8      3.64   3.16  0  0.003194
9     -0.33   0.59  0  0.041583
10     0.99  -0.29  0  0.210057
11     0.54  -1.73  0  0.005637
12     2.38   2.53  0  0.003981
13     0.29   1.20  0  0.019361
14    -0.10  -2.05  0  0.005793
15     1.75   2.02  0  0.004671
16     1.46   0.20  0  0.015456
17     1.18   1.41  0  0.012129
18     4.24   7.04  0  0.026276
19     2.55   4.78  0  0.000706
20     3.88   5.02  0  0.007200
21     1.25   0.10  0  0.004430
22    -0.04  -1.86  0  0.014199
23     0.21  -1.28  0  0.008723
24     1.47   2.67  0  0.001721
25     0.19  -0.50  0  0.002515
26     3.35   4.83  0  0.001132
27     0.85   1.65  0  0.034738
28     1.25   1.73  0  0.011902
29    -1.25  -2.32  0  0.011916
...     ...    ... ..       ...
9697   1.32  -0.51  0  0.012129
9698  18.54  29.52  0  0.002148
9699   6.44  14.87  0  0.001752
9700   7.89  17.30  0  0.015465
9701   5.54   2.77  0  0.005739
9702  19.95  27.43  0  0.044459
9703  14.70  18.95  0  0.003884
9704  11.87  15.91  0  0.001148
9705  12.53  17.22  0  0.000774
9706  23.01  40.47  0  0.006556
9707   1.07   4.48  0  0.009148
9708   5.22  15.00  0  0.009225
9709   7.10   5.91  0  0.002634
9710  20.89  41.20  0  0.005612
9711   7.00   6.21  0  0.003837
9712  13.37  19.81  0  0.020465
9713  25.02  43.40  0  0.001434
9714  12.01  15.54  0  0.001645
9715  14.62  20.51  0  0.022678
9716  16.25  17.66  0  0.013341
9717  11.50  16.02  1  0.007942
9718  23.11  36.73  1  0.016428
9719  12.96  14.72  1  0.003055
9720  10.03  21.81  1  0.640396
9721   6.25  14.55  1  0.411098
9722   7.91   8.83  1  0.072823
9723  10.03  21.81  1  0.654751
9724  12.19  23.90  1  0.015545
9725  17.06  25.58  1  0.932408
9726  11.39  18.65  1  0.980807

[9727 rows x 4 columns]
>>> data
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E  \
0     0.000000    1.0  0.250000    0.0   
1     0.000000    0.0  0.000000    0.0   
2     0.250000    0.0  0.000000    0.0   
3     0.250000    0.0  0.000000    0.0   
4     0.250000    0.0  0.000000    0.0   
5     0.000000    0.0  0.000000    0.0   
6     0.000000    0.0  0.000000    0.0   
7     0.250000    0.0  0.000000    0.0   
8     0.000000    0.0  0.000000    0.0   
9     0.000000    0.0  0.000000    1.0   
10    0.000000    0.0  0.000000    0.0   
11    0.000000    0.0  0.000000    0.0   
12    0.000000    0.0  0.000000    0.0   
13    0.000000    0.0  0.000000    0.0   
14    0.000000    0.0  0.000000    0.0   
15    0.000000    0.0  0.000000    0.0   
16    0.000000    0.0  0.000000    0.0   
17    0.000000    0.0  0.000000    0.0   
18    0.000000    0.0  0.000000    0.0   
19    0.000000    0.0  0.000000    1.0   
20    0.000000    0.0  0.000000    1.0   
21    0.000000    0.0  0.000000    0.0   
22    0.000000    0.0  0.000000    0.0   
23    0.000000    0.0  0.000000    0.0   
24    0.250000    0.0  0.000000    1.0   
25    0.250000    0.0  0.000000    0.0   
26    0.250000    0.0  0.000000    0.0   
27    0.250000    0.0  0.000000    0.0   
28    0.250000    0.0  0.000000    0.0   
29    0.250000    0.0  0.000000    0.0   
...        ...    ...       ...    ...   
9697  0.300000    0.0  0.000000    0.0   
9698  0.033333    9.0  0.300000    2.0   
9699  0.033333    0.0  0.000000    3.0   
9700  0.066667    2.0  0.066667    3.0   
9701  0.066667    0.0  0.000000    1.0   
9702  0.033333    2.0  0.066667    0.0   
9703  0.033333   11.0  0.366667    0.0   
9704  0.066667    6.0  0.200000    0.0   
9705  0.000000    6.0  0.200000    1.0   
9706  0.133333    0.0  0.000000    5.0   
9707  0.166667    0.0  0.000000    3.0   
9708  0.100000    0.0  0.000000    2.0   
9709  0.000000    0.0  0.000000    2.0   
9710  0.100000    0.0  0.000000    8.0   
9711  0.066667    4.0  0.133333    0.0   
9712  0.200000    0.0  0.000000    0.0   
9713  0.066667    0.0  0.000000    3.0   
9714  0.100000    0.0  0.000000    1.0   
9715  0.166667    0.0  0.000000    3.0   
9716  0.000000    6.0  0.200000    2.0   
9717  0.100000    0.0  0.000000    0.0   
9718  0.000000    2.0  0.066667    1.0   
9719  0.066667    0.0  0.000000    0.0   
9720  0.066667    0.0  0.000000    0.0   
9721  0.000000    1.0  0.033333    0.0   
9722  0.033333    1.0  0.033333    0.0   
9723  0.066667    0.0  0.000000    0.0   
9724  0.466667    0.0  0.000000    7.0   
9725  0.400000    1.0  0.033333    2.0   
9726  0.400000    1.0  0.033333    2.0   

                            ...                             per_R  num_T  \
0                           ...                          0.000000    0.0   
1                           ...                          0.000000    0.0   
2                           ...                          0.000000    0.0   
3                           ...                          0.000000    1.0   
4                           ...                          0.000000    0.0   
5                           ...                          0.250000    0.0   
6                           ...                          0.250000    0.0   
7                           ...                          0.000000    0.0   
8                           ...                          0.000000    0.0   
9                           ...                          0.000000    0.0   
10                          ...                          0.000000    1.0   
11                          ...                          0.000000    0.0   
12                          ...                          0.000000    0.0   
13                          ...                          0.000000    1.0   
14                          ...                          0.000000    0.0   
15                          ...                          0.250000    0.0   
16                          ...                          0.250000    0.0   
17                          ...                          0.250000    0.0   
18                          ...                          0.000000    0.0   
19                          ...                          0.000000    0.0   
20                          ...                          0.000000    0.0   
21                          ...                          0.000000    0.0   
22                          ...                          0.250000    0.0   
23                          ...                          0.250000    0.0   
24                          ...                          0.000000    1.0   
25                          ...                          0.000000    1.0   
26                          ...                          0.000000    0.0   
27                          ...                          0.250000    0.0   
28                          ...                          0.000000    0.0   
29                          ...                          0.000000    0.0   
...                         ...                               ...    ...   
9697                        ...                          0.000000    2.0   
9698                        ...                          0.000000    3.0   
9699                        ...                          0.033333    1.0   
9700                        ...                          0.133333    0.0   
9701                        ...                          0.000000    2.0   
9702                        ...                          0.033333    2.0   
9703                        ...                          0.000000    3.0   
9704                        ...                          0.033333    1.0   
9705                        ...                          0.100000    0.0   
9706                        ...                          0.166667    3.0   
9707                        ...                          0.033333    1.0   
9708                        ...                          0.066667    2.0   
9709                        ...                          0.066667    2.0   
9710                        ...                          0.100000    1.0   
9711                        ...                          0.000000    3.0   
9712                        ...                          0.033333    1.0   
9713                        ...                          0.033333    1.0   
9714                        ...                          0.000000    1.0   
9715                        ...                          0.133333    2.0   
9716                        ...                          0.066667    1.0   
9717                        ...                          0.066667    3.0   
9718                        ...                          0.200000    1.0   
9719                        ...                          0.033333    2.0   
9720                        ...                          0.133333    0.0   
9721                        ...                          0.300000    0.0   
9722                        ...                          0.033333    0.0   
9723                        ...                          0.133333    0.0   
9724                        ...                          0.000000    0.0   
9725                        ...                          0.000000    0.0   
9726                        ...                          0.233333    0.0   

         per_T  num_W     per_W  num_V     per_V  num_Y     per_Y  \
0     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
2     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
4     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
7     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
8     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
9     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
10    0.250000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
11    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
13    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
14    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
16    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
20    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
22    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
23    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
24    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
25    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
26    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
27    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
28    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
29    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
...        ...    ...       ...    ...       ...    ...       ...   
9697  0.066667    1.0  0.033333    5.0  0.166667    2.0  0.066667   
9698  0.100000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9699  0.033333    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9700  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9701  0.066667    3.0  0.100000    1.0  0.033333    1.0  0.033333   
9702  0.066667    0.0  0.000000    0.0  0.000000    2.0  0.066667   
9703  0.100000    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9704  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9705  0.000000    0.0  0.000000    2.0  0.066667    1.0  0.033333   
9706  0.100000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9707  0.033333    1.0  0.033333    4.0  0.133333    0.0  0.000000   
9708  0.066667    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9709  0.066667    3.0  0.100000    3.0  0.100000    1.0  0.033333   
9710  0.033333    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9711  0.100000    1.0  0.033333    2.0  0.066667    2.0  0.066667   
9712  0.033333    1.0  0.033333    3.0  0.100000    0.0  0.000000   
9713  0.033333    0.0  0.000000    2.0  0.066667    0.0  0.000000   
9714  0.033333    1.0  0.033333    1.0  0.033333    2.0  0.066667   
9715  0.066667    0.0  0.000000    1.0  0.033333    4.0  0.133333   
9716  0.033333    4.0  0.133333    1.0  0.033333    1.0  0.033333   
9717  0.100000    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9718  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9719  0.066667    1.0  0.033333    1.0  0.033333    1.0  0.033333   
9720  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9721  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9722  0.000000    2.0  0.066667    4.0  0.133333    0.0  0.000000   
9723  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9724  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9725  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9726  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   

                                                      X  
0     [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...  
1     [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...  
2     [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....  
3     [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...  
4     [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...  
5     [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...  
6     [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...  
7     [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....  
8     [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...  
9     [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...  
10    [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...  
11    [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...  
12    [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...  
13    [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...  
14    [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...  
15    [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...  
16    [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...  
17    [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...  
18    [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...  
19    [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...  
20    [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...  
21    [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...  
22    [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...  
23    [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...  
24    [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...  
25    [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...  
26    [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....  
27    [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...  
28    [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...  
29    [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...  
...                                                 ...  
9697  [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...  
9698  [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...  
9699  [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...  
9700  [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....  
9701  [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...  
9702  [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...  
9703  [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...  
9704  [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...  
9705  [[14.0, 1.0, 14.0, 4.0, 15.0, 3.0, 14.0, 15.0,...  
9706  [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....  
9707  [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...  
9708  [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...  
9709  [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...  
9710  [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...  
9711  [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....  
9712  [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...  
9713  [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...  
9714  [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...  
9715  [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...  
9716  [[19.0, 1.0, 12.0, 8.0, 17.0, 9.0, 17.0, 16.0,...  
9717  [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...  
9718  [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....  
9719  [[4.0, 17.0, 16.0, 10.0, 11.0, 14.0, 0.0, 4.0,...  
9720  [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...  
9721  [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...  
9722  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...  
9723  [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...  
9724  [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...  
9725  [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...  
9726  [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...  

[9727 rows x 46 columns]
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[data_xgb.keys()[-5:-1]],on='seq')

Traceback (most recent call last):
  File "<pyshell#30>", line 1, in <module>
    data.merge(data_xgb[data_xgb.keys()[-5:-1]],on='seq')
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'seq'
>>> data.seq
0                                 SPCG
1                                 VPSG
2                                 ADKP
3                                 AGLT
4                                 APGW
5                                 FLRN
6                                 FYRI
7                                 GFAD
8                                 GSWD
9                                 ILME
10                                LWKT
11                                LWSG
12                                LYDN
13                                NTQM
14                                PGLW
15                                QGRF
16                                RGMW
17                                RSFN
18                                SDKP
19                                VGSE
20                                WGHE
21                                YDFI
22                                YLRF
23                                YMRF
24                                AETF
25                                AFTS
26                                AGDV
27                                AIRS
28                                AKPF
29                                ALPF
                     ...              
9697    PAIYIGATVGPSVWAYLVALVGAAAVTAAN
9698    PCEKCTSGCKCPSKDECAKTCSKPCSCCPT
9699    PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV
9700    QQARQNLQNLYINRCLREICQELKEIRAML
9701    QTNWQKLEVFWAKHMWNFISGIQYLAGLST
9702    RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK
9703    SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ
9704    SCNNSCQSHSDCASHCICTFRGCGAVNGLP
9705    SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE
9706    SEKAAEEAYTRTTRALHERFDRLERMLDDN
9707    TAFVEPFVILLILIANAIVGVWQERNAENA
9708    TPVERQTIYSQAPSLNPNLILAAPPKERNQ
9709    TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR
9710    VEKLTADAELQRLKNERHEEAELERLLSEY
9711    VIHCDAATICPDGTTCSLSPYGVWYCSPFS
9712    VKGRIDAPDFPSSPAILGKAATDVVAAWKS
9713    VTVDDDDDDNDPENRIAKKMLLEEIKANLS
9714    YAEGTFISDYSIAMDKIHQQDFVNWLLAQK
9715    YAEKVAQEKGFLYRLTSRYRHYAAFERATF
9716    YCQKWMWTCDSERKCCEGMVCRLWCKKKLW
9717    ADVFDRGGPYLQRGVADLVPTATLLDTYSP
9718    CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ
9719    GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS
9720    LLIILRRRIRKQAHAHSKNHQQQNPHQPPM
9721    MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG
9722    MVKSKIGSWILVLFVAMWSDVGLCKKRPKP
9723    NHQQQNPHQPPMLLIILRRRIRKQAHAHSK
9724    WEAALAEALAEALAEHLAEALAEALEALAA
9725    WEAKLAKALAKALAKHLAKALAKALKACEA
9726    WEARLARALARALARHLARALARALRACEA
Name: seq, Length: 9727, dtype: object
>>> data_xgb.seq
0                                 SPCG
1                                 VPSG
2                                 ADKP
3                                 AGLT
4                                 APGW
5                                 FLRN
6                                 FYRI
7                                 GFAD
8                                 GSWD
9                                 ILME
10                                LWKT
11                                LWSG
12                                LYDN
13                                NTQM
14                                PGLW
15                                QGRF
16                                RGMW
17                                RSFN
18                                SDKP
19                                VGSE
20                                WGHE
21                                YDFI
22                                YLRF
23                                YMRF
24                                AETF
25                                AFTS
26                                AGDV
27                                AIRS
28                                AKPF
29                                ALPF
                     ...              
9697    PAIYIGATVGPSVWAYLVALVGAAAVTAAN
9698    PCEKCTSGCKCPSKDECAKTCSKPCSCCPT
9699    PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV
9700    QQARQNLQNLYINRCLREICQELKEIRAML
9701    QTNWQKLEVFWAKHMWNFISGIQYLAGLST
9702    RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK
9703    SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ
9704    SCNNSCQSHSDCASHCICTFRGCGAVNGLP
9705    SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE
9706    SEKAAEEAYTRTTRALHERFDRLERMLDDN
9707    TAFVEPFVILLILIANAIVGVWQERNAENA
9708    TPVERQTIYSQAPSLNPNLILAAPPKERNQ
9709    TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR
9710    VEKLTADAELQRLKNERHEEAELERLLSEY
9711    VIHCDAATICPDGTTCSLSPYGVWYCSPFS
9712    VKGRIDAPDFPSSPAILGKAATDVVAAWKS
9713    VTVDDDDDDNDPENRIAKKMLLEEIKANLS
9714    YAEGTFISDYSIAMDKIHQQDFVNWLLAQK
9715    YAEKVAQEKGFLYRLTSRYRHYAAFERATF
9716    YCQKWMWTCDSERKCCEGMVCRLWCKKKLW
9717    ADVFDRGGPYLQRGVADLVPTATLLDTYSP
9718    CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ
9719    GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS
9720    LLIILRRRIRKQAHAHSKNHQQQNPHQPPM
9721    MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG
9722    MVKSKIGSWILVLFVAMWSDVGLCKKRPKP
9723    NHQQQNPHQPPMLLIILRRRIRKQAHAHSK
9724    WEAALAEALAEALAEHLAEALAEALEALAA
9725    WEAKLAKALAKALAKHLAKALAKALKACEA
9726    WEARLARALARALARHLARALARALRACEA
Name: seq, Length: 9727, dtype: object
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[data_xgb.keys()[-5:-1]],on=['seq'])

Traceback (most recent call last):
  File "<pyshell#33>", line 1, in <module>
    data.merge(data_xgb[data_xgb.keys()[-5:-1]],on=['seq'])
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'seq'
>>> data.merge(data_xgb[data_xgb.keys()[-5:-1]])
KeyboardInterrupt
>>> data.merge(data_xgb[data_xgb.keys()[-5:-1]+['seq',]],on='seq')

Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    data.merge(data_xgb[data_xgb.keys()[-5:-1]+['seq',]],on='seq')
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1958, in __getitem__
    return self._getitem_array(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 2002, in _getitem_array
    indexer = self.loc._convert_to_indexer(key, axis=1)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexing.py", line 1231, in _convert_to_indexer
    raise KeyError('%s not in index' % objarr[mask])
KeyError: "Index([u'Gwifseq', u'Goctseq', u'yseq', u'testPredseq'], dtype='object') not in index"
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[list(data_xgb.keys()[-5:-1])+['seq',]],on='seq')
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30    3.0   
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30    2.0   
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E    ...        per_W  num_V     per_V  \
0     0.000000    1.0  0.250000    0.0    ...     0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
2     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
4     0.250000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
7     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
8     0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
9     0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
10    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
11    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
13    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
14    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
16    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0    ...     0.000000    1.0  0.250000   
20    0.000000    0.0  0.000000    1.0    ...     0.250000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
22    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
23    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
24    0.250000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
25    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
26    0.250000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
27    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
28    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
29    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
...        ...    ...       ...    ...    ...          ...    ...       ...   
9843  0.033333    1.0  0.033333    0.0    ...     0.066667    4.0  0.133333   
9844  0.033333    1.0  0.033333    0.0    ...     0.066667    4.0  0.133333   
9845  0.100000    0.0  0.000000    1.0    ...     0.000000    2.0  0.066667   
9846  0.066667    0.0  0.000000    4.0    ...     0.033333    1.0  0.033333   
9847  0.300000    0.0  0.000000    0.0    ...     0.033333    5.0  0.166667   
9848  0.033333    9.0  0.300000    2.0    ...     0.000000    0.0  0.000000   
9849  0.033333    0.0  0.000000    3.0    ...     0.000000    3.0  0.100000   
9850  0.066667    2.0  0.066667    3.0    ...     0.000000    0.0  0.000000   
9851  0.066667    0.0  0.000000    1.0    ...     0.100000    1.0  0.033333   
9852  0.033333    2.0  0.066667    0.0    ...     0.000000    0.0  0.000000   
9853  0.033333   11.0  0.366667    0.0    ...     0.000000    1.0  0.033333   
9854  0.066667    6.0  0.200000    0.0    ...     0.000000    1.0  0.033333   
9855  0.133333    0.0  0.000000    5.0    ...     0.000000    0.0  0.000000   
9856  0.166667    0.0  0.000000    3.0    ...     0.033333    4.0  0.133333   
9857  0.100000    0.0  0.000000    2.0    ...     0.000000    1.0  0.033333   
9858  0.000000    0.0  0.000000    2.0    ...     0.100000    3.0  0.100000   
9859  0.100000    0.0  0.000000    8.0    ...     0.000000    1.0  0.033333   
9860  0.066667    4.0  0.133333    0.0    ...     0.033333    2.0  0.066667   
9861  0.200000    0.0  0.000000    0.0    ...     0.033333    3.0  0.100000   
9862  0.066667    0.0  0.000000    3.0    ...     0.000000    2.0  0.066667   
9863  0.100000    0.0  0.000000    1.0    ...     0.033333    1.0  0.033333   
9864  0.166667    0.0  0.000000    3.0    ...     0.000000    1.0  0.033333   
9865  0.100000    0.0  0.000000    0.0    ...     0.000000    3.0  0.100000   
9866  0.000000    2.0  0.066667    1.0    ...     0.000000    1.0  0.033333   
9867  0.066667    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
9868  0.000000    1.0  0.033333    0.0    ...     0.000000    0.0  0.000000   
9869  0.066667    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
9870  0.466667    0.0  0.000000    7.0    ...     0.033333    0.0  0.000000   
9871  0.400000    1.0  0.033333    2.0    ...     0.033333    0.0  0.000000   
9872  0.400000    1.0  0.033333    2.0    ...     0.033333    0.0  0.000000   

      num_Y     per_Y                                                  X  \
0       0.0  0.000000  [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...   
1       0.0  0.000000  [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...   
2       0.0  0.000000  [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....   
3       0.0  0.000000  [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...   
4       0.0  0.000000  [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...   
5       0.0  0.000000  [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...   
6       1.0  0.250000  [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...   
7       0.0  0.000000  [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....   
8       0.0  0.000000  [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...   
9       0.0  0.000000  [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...   
10      0.0  0.000000  [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...   
11      0.0  0.000000  [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...   
12      1.0  0.250000  [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...   
13      0.0  0.000000  [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...   
14      0.0  0.000000  [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...   
15      0.0  0.000000  [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...   
16      0.0  0.000000  [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...   
17      0.0  0.000000  [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...   
18      0.0  0.000000  [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...   
19      0.0  0.000000  [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...   
20      0.0  0.000000  [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...   
21      1.0  0.250000  [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...   
22      1.0  0.250000  [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...   
23      1.0  0.250000  [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...   
24      0.0  0.000000  [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...   
25      0.0  0.000000  [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...   
26      0.0  0.000000  [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....   
27      0.0  0.000000  [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...   
28      0.0  0.000000  [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...   
29      0.0  0.000000  [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...   
...     ...       ...                                                ...   
9843    0.0  0.000000  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...   
9844    0.0  0.000000  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...   
9845    1.0  0.033333  [[11.0, 11.0, 3.0, 6.0, 16.0, 0.0, 2.0, 11.0, ...   
9846    1.0  0.033333  [[11.0, 16.0, 10.0, 3.0, 16.0, 10.0, 19.0, 8.0...   
9847    2.0  0.066667  [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...   
9848    0.0  0.000000  [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...   
9849    2.0  0.066667  [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...   
9850    1.0  0.033333  [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....   
9851    1.0  0.033333  [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...   
9852    2.0  0.066667  [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...   
9853    0.0  0.000000  [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...   
9854    0.0  0.000000  [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...   
9855    1.0  0.033333  [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....   
9856    0.0  0.000000  [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...   
9857    1.0  0.033333  [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...   
9858    1.0  0.033333  [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...   
9859    1.0  0.033333  [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...   
9860    2.0  0.066667  [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....   
9861    0.0  0.000000  [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...   
9862    0.0  0.000000  [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...   
9863    2.0  0.066667  [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...   
9864    4.0  0.133333  [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...   
9865    2.0  0.066667  [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...   
9866    0.0  0.000000  [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....   
9867    0.0  0.000000  [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...   
9868    1.0  0.033333  [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...   
9869    0.0  0.000000  [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...   
9870    0.0  0.000000  [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...   
9871    0.0  0.000000  [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...   
9872    0.0  0.000000  [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...   

       Gwif   Goct  y  testPred  
0      1.38   1.73  0  0.007797  
1      0.63   1.29  0  0.002599  
2      4.24   7.08  0  0.031334  
3      0.89   0.65  0  0.002652  
4      0.92  -0.30  0  0.001054  
5      0.16  -0.30  0  0.020269  
6     -0.16  -1.73  0  0.039302  
7      3.30   3.58  0  0.000825  
8      3.64   3.16  0  0.003194  
9     -0.33   0.59  0  0.041583  
10     0.99  -0.29  0  0.210057  
11     0.54  -1.73  0  0.005637  
12     2.38   2.53  0  0.003981  
13     0.29   1.20  0  0.019361  
14    -0.10  -2.05  0  0.005793  
15     1.75   2.02  0  0.004671  
16     1.46   0.20  0  0.015456  
17     1.18   1.41  0  0.012129  
18     4.24   7.04  0  0.026276  
19     2.55   4.78  0  0.000706  
20     3.88   5.02  0  0.007200  
21     1.25   0.10  0  0.004430  
22    -0.04  -1.86  0  0.014199  
23     0.21  -1.28  0  0.008723  
24     1.47   2.67  0  0.001721  
25     0.19  -0.50  0  0.002515  
26     3.35   4.83  0  0.001132  
27     0.85   1.65  0  0.034738  
28     1.25   1.73  0  0.011902  
29    -1.25  -2.32  0  0.011916  
...     ...    ... ..       ...  
9843   7.91   8.83  0  0.643268  
9844   7.91   8.83  1  0.072823  
9845  14.02  23.74  0  0.008374  
9846  12.61  23.45  0  0.004381  
9847   1.32  -0.51  0  0.012129  
9848  18.54  29.52  0  0.002148  
9849   6.44  14.87  0  0.001752  
9850   7.89  17.30  0  0.015465  
9851   5.54   2.77  0  0.005739  
9852  19.95  27.43  0  0.044459  
9853  14.70  18.95  0  0.003884  
9854  11.87  15.91  0  0.001148  
9855  23.01  40.47  0  0.006556  
9856   1.07   4.48  0  0.009148  
9857   5.22  15.00  0  0.009225  
9858   7.10   5.91  0  0.002634  
9859  20.89  41.20  0  0.005612  
9860   7.00   6.21  0  0.003837  
9861  13.37  19.81  0  0.020465  
9862  25.02  43.40  0  0.001434  
9863  12.01  15.54  0  0.001645  
9864  14.62  20.51  0  0.022678  
9865  11.50  16.02  1  0.007942  
9866  23.11  36.73  1  0.016428  
9867  10.03  21.81  1  0.640396  
9868   6.25  14.55  1  0.411098  
9869  10.03  21.81  1  0.654751  
9870  12.19  23.90  1  0.015545  
9871  17.06  25.58  1  0.932408  
9872  11.39  18.65  1  0.980807  

[9873 rows x 50 columns]
>>> x= data.merge(data_xgb[list(data_xgb.keys()[-5:-1])+['seq',]],on='seq')
>>> x
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30    3.0   
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30    2.0   
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E    ...        per_W  num_V     per_V  \
0     0.000000    1.0  0.250000    0.0    ...     0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
2     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
4     0.250000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
7     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
8     0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
9     0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
10    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
11    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
13    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
14    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
16    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0    ...     0.000000    1.0  0.250000   
20    0.000000    0.0  0.000000    1.0    ...     0.250000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
22    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
23    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
24    0.250000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
25    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
26    0.250000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
27    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
28    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
29    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
...        ...    ...       ...    ...    ...          ...    ...       ...   
9843  0.033333    1.0  0.033333    0.0    ...     0.066667    4.0  0.133333   
9844  0.033333    1.0  0.033333    0.0    ...     0.066667    4.0  0.133333   
9845  0.100000    0.0  0.000000    1.0    ...     0.000000    2.0  0.066667   
9846  0.066667    0.0  0.000000    4.0    ...     0.033333    1.0  0.033333   
9847  0.300000    0.0  0.000000    0.0    ...     0.033333    5.0  0.166667   
9848  0.033333    9.0  0.300000    2.0    ...     0.000000    0.0  0.000000   
9849  0.033333    0.0  0.000000    3.0    ...     0.000000    3.0  0.100000   
9850  0.066667    2.0  0.066667    3.0    ...     0.000000    0.0  0.000000   
9851  0.066667    0.0  0.000000    1.0    ...     0.100000    1.0  0.033333   
9852  0.033333    2.0  0.066667    0.0    ...     0.000000    0.0  0.000000   
9853  0.033333   11.0  0.366667    0.0    ...     0.000000    1.0  0.033333   
9854  0.066667    6.0  0.200000    0.0    ...     0.000000    1.0  0.033333   
9855  0.133333    0.0  0.000000    5.0    ...     0.000000    0.0  0.000000   
9856  0.166667    0.0  0.000000    3.0    ...     0.033333    4.0  0.133333   
9857  0.100000    0.0  0.000000    2.0    ...     0.000000    1.0  0.033333   
9858  0.000000    0.0  0.000000    2.0    ...     0.100000    3.0  0.100000   
9859  0.100000    0.0  0.000000    8.0    ...     0.000000    1.0  0.033333   
9860  0.066667    4.0  0.133333    0.0    ...     0.033333    2.0  0.066667   
9861  0.200000    0.0  0.000000    0.0    ...     0.033333    3.0  0.100000   
9862  0.066667    0.0  0.000000    3.0    ...     0.000000    2.0  0.066667   
9863  0.100000    0.0  0.000000    1.0    ...     0.033333    1.0  0.033333   
9864  0.166667    0.0  0.000000    3.0    ...     0.000000    1.0  0.033333   
9865  0.100000    0.0  0.000000    0.0    ...     0.000000    3.0  0.100000   
9866  0.000000    2.0  0.066667    1.0    ...     0.000000    1.0  0.033333   
9867  0.066667    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
9868  0.000000    1.0  0.033333    0.0    ...     0.000000    0.0  0.000000   
9869  0.066667    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
9870  0.466667    0.0  0.000000    7.0    ...     0.033333    0.0  0.000000   
9871  0.400000    1.0  0.033333    2.0    ...     0.033333    0.0  0.000000   
9872  0.400000    1.0  0.033333    2.0    ...     0.033333    0.0  0.000000   

      num_Y     per_Y                                                  X  \
0       0.0  0.000000  [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...   
1       0.0  0.000000  [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...   
2       0.0  0.000000  [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....   
3       0.0  0.000000  [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...   
4       0.0  0.000000  [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...   
5       0.0  0.000000  [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...   
6       1.0  0.250000  [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...   
7       0.0  0.000000  [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....   
8       0.0  0.000000  [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...   
9       0.0  0.000000  [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...   
10      0.0  0.000000  [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...   
11      0.0  0.000000  [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...   
12      1.0  0.250000  [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...   
13      0.0  0.000000  [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...   
14      0.0  0.000000  [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...   
15      0.0  0.000000  [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...   
16      0.0  0.000000  [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...   
17      0.0  0.000000  [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...   
18      0.0  0.000000  [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...   
19      0.0  0.000000  [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...   
20      0.0  0.000000  [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...   
21      1.0  0.250000  [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...   
22      1.0  0.250000  [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...   
23      1.0  0.250000  [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...   
24      0.0  0.000000  [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...   
25      0.0  0.000000  [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...   
26      0.0  0.000000  [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....   
27      0.0  0.000000  [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...   
28      0.0  0.000000  [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...   
29      0.0  0.000000  [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...   
...     ...       ...                                                ...   
9843    0.0  0.000000  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...   
9844    0.0  0.000000  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...   
9845    1.0  0.033333  [[11.0, 11.0, 3.0, 6.0, 16.0, 0.0, 2.0, 11.0, ...   
9846    1.0  0.033333  [[11.0, 16.0, 10.0, 3.0, 16.0, 10.0, 19.0, 8.0...   
9847    2.0  0.066667  [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...   
9848    0.0  0.000000  [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...   
9849    2.0  0.066667  [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...   
9850    1.0  0.033333  [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....   
9851    1.0  0.033333  [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...   
9852    2.0  0.066667  [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...   
9853    0.0  0.000000  [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...   
9854    0.0  0.000000  [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...   
9855    1.0  0.033333  [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....   
9856    0.0  0.000000  [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...   
9857    1.0  0.033333  [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...   
9858    1.0  0.033333  [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...   
9859    1.0  0.033333  [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...   
9860    2.0  0.066667  [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....   
9861    0.0  0.000000  [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...   
9862    0.0  0.000000  [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...   
9863    2.0  0.066667  [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...   
9864    4.0  0.133333  [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...   
9865    2.0  0.066667  [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...   
9866    0.0  0.000000  [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....   
9867    0.0  0.000000  [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...   
9868    1.0  0.033333  [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...   
9869    0.0  0.000000  [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...   
9870    0.0  0.000000  [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...   
9871    0.0  0.000000  [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...   
9872    0.0  0.000000  [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...   

       Gwif   Goct  y  testPred  
0      1.38   1.73  0  0.007797  
1      0.63   1.29  0  0.002599  
2      4.24   7.08  0  0.031334  
3      0.89   0.65  0  0.002652  
4      0.92  -0.30  0  0.001054  
5      0.16  -0.30  0  0.020269  
6     -0.16  -1.73  0  0.039302  
7      3.30   3.58  0  0.000825  
8      3.64   3.16  0  0.003194  
9     -0.33   0.59  0  0.041583  
10     0.99  -0.29  0  0.210057  
11     0.54  -1.73  0  0.005637  
12     2.38   2.53  0  0.003981  
13     0.29   1.20  0  0.019361  
14    -0.10  -2.05  0  0.005793  
15     1.75   2.02  0  0.004671  
16     1.46   0.20  0  0.015456  
17     1.18   1.41  0  0.012129  
18     4.24   7.04  0  0.026276  
19     2.55   4.78  0  0.000706  
20     3.88   5.02  0  0.007200  
21     1.25   0.10  0  0.004430  
22    -0.04  -1.86  0  0.014199  
23     0.21  -1.28  0  0.008723  
24     1.47   2.67  0  0.001721  
25     0.19  -0.50  0  0.002515  
26     3.35   4.83  0  0.001132  
27     0.85   1.65  0  0.034738  
28     1.25   1.73  0  0.011902  
29    -1.25  -2.32  0  0.011916  
...     ...    ... ..       ...  
9843   7.91   8.83  0  0.643268  
9844   7.91   8.83  1  0.072823  
9845  14.02  23.74  0  0.008374  
9846  12.61  23.45  0  0.004381  
9847   1.32  -0.51  0  0.012129  
9848  18.54  29.52  0  0.002148  
9849   6.44  14.87  0  0.001752  
9850   7.89  17.30  0  0.015465  
9851   5.54   2.77  0  0.005739  
9852  19.95  27.43  0  0.044459  
9853  14.70  18.95  0  0.003884  
9854  11.87  15.91  0  0.001148  
9855  23.01  40.47  0  0.006556  
9856   1.07   4.48  0  0.009148  
9857   5.22  15.00  0  0.009225  
9858   7.10   5.91  0  0.002634  
9859  20.89  41.20  0  0.005612  
9860   7.00   6.21  0  0.003837  
9861  13.37  19.81  0  0.020465  
9862  25.02  43.40  0  0.001434  
9863  12.01  15.54  0  0.001645  
9864  14.62  20.51  0  0.022678  
9865  11.50  16.02  1  0.007942  
9866  23.11  36.73  1  0.016428  
9867  10.03  21.81  1  0.640396  
9868   6.25  14.55  1  0.411098  
9869  10.03  21.81  1  0.654751  
9870  12.19  23.90  1  0.015545  
9871  17.06  25.58  1  0.932408  
9872  11.39  18.65  1  0.980807  

[9873 rows x 50 columns]
>>> x['X']
0       [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...
1       [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...
2       [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....
3       [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...
4       [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...
5       [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...
6       [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...
7       [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....
8       [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...
9       [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...
10      [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...
11      [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...
12      [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...
13      [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...
14      [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...
15      [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...
16      [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...
17      [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...
18      [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...
19      [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...
20      [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...
21      [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...
22      [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...
23      [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...
24      [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...
25      [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...
26      [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....
27      [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...
28      [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...
29      [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...
                              ...                        
9843    [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...
9844    [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...
9845    [[11.0, 11.0, 3.0, 6.0, 16.0, 0.0, 2.0, 11.0, ...
9846    [[11.0, 16.0, 10.0, 3.0, 16.0, 10.0, 19.0, 8.0...
9847    [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...
9848    [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...
9849    [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...
9850    [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....
9851    [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...
9852    [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...
9853    [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...
9854    [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...
9855    [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....
9856    [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...
9857    [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...
9858    [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...
9859    [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...
9860    [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....
9861    [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...
9862    [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...
9863    [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...
9864    [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...
9865    [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...
9866    [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....
9867    [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...
9868    [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...
9869    [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...
9870    [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...
9871    [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...
9872    [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...
Name: X, Length: 9873, dtype: object
>>> x['X'].iloc[0]
[array([14.000, 13.000, 1.000, 4.000]), array([[0.130, 0.450, -0.240, 0.010],
       [0.460, 0.140, -0.020, 1.150],
       [0.330, -0.310, 0.220, 1.140],
       [0.000, 0.000, 0.000, 0.000]]), [4, 0.0, 1.3799999999999999, 1.73, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'SPCG', 0]
>>> x[['X','].iloc[0]
KeyboardInterrupt
>>> data_xgb.keys()
Index([u'seq', u'source', u'size', u'type_aa', u'len', u'X', u'num_A',
       u'per_A', u'num_C', u'per_C', u'num_E', u'per_E', u'num_D', u'per_D',
       u'num_G', u'per_G', u'num_F', u'per_F', u'num_I', u'per_I', u'num_H',
       u'per_H', u'num_K', u'per_K', u'num_M', u'per_M', u'num_L', u'per_L',
       u'num_N', u'per_N', u'num_Q', u'per_Q', u'num_P', u'per_P', u'num_S',
       u'per_S', u'num_R', u'per_R', u'num_T', u'per_T', u'num_W', u'per_W',
       u'num_V', u'per_V', u'num_Y', u'per_Y', u'length', u'netcharge',
       u'Gwif', u'Goct', u'y', u'testPred', u'testY'],
      dtype='object')
>>> x[['X','Gwif','Goct']].iloc[0]
X       [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...
Gwif                                                 1.38
Goct                                                 1.73
Name: 0, dtype: object
>>> x[['X','Gwif','Goct']].iloc[0]
X       [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...
Gwif                                                 1.38
Goct                                                 1.73
Name: 0, dtype: object
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9873
train+val  size : 7898
size of different sets: 11125 1975 1975
0.831335 0.807089
0.918009462405 0.808129137031 0.865237333333
0.937701 0.918481
0.958502782752 0.838064794639 0.894656
0.947493 0.918481
0.975677535865 0.844509312174 0.898362666667
SAVED

0.916934 0.885063
0.984411526168 0.853944825885 0.915808
SAVED

0.965558 0.936709
0.990928274461 0.848672038811 0.911936
SAVED

0.970454 0.938734
0.99484146516 0.859083994573 0.907109333333
SAVED

0.980922 0.947848
0.996814350926 0.871382025244 0.920618666667
SAVED

0.978896 0.948354
0.997694033402 0.86324672121 0.90784
SAVED


 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 303, in <module>
    dropout : 0.2,learning_rate : lr}) #sgd
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
>>> data
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E  \
0     0.000000    1.0  0.250000    0.0   
1     0.000000    0.0  0.000000    0.0   
2     0.250000    0.0  0.000000    0.0   
3     0.250000    0.0  0.000000    0.0   
4     0.250000    0.0  0.000000    0.0   
5     0.000000    0.0  0.000000    0.0   
6     0.000000    0.0  0.000000    0.0   
7     0.250000    0.0  0.000000    0.0   
8     0.000000    0.0  0.000000    0.0   
9     0.000000    0.0  0.000000    1.0   
10    0.000000    0.0  0.000000    0.0   
11    0.000000    0.0  0.000000    0.0   
12    0.000000    0.0  0.000000    0.0   
13    0.000000    0.0  0.000000    0.0   
14    0.000000    0.0  0.000000    0.0   
15    0.000000    0.0  0.000000    0.0   
16    0.000000    0.0  0.000000    0.0   
17    0.000000    0.0  0.000000    0.0   
18    0.000000    0.0  0.000000    0.0   
19    0.000000    0.0  0.000000    1.0   
20    0.000000    0.0  0.000000    1.0   
21    0.000000    0.0  0.000000    0.0   
22    0.000000    0.0  0.000000    0.0   
23    0.000000    0.0  0.000000    0.0   
24    0.250000    0.0  0.000000    1.0   
25    0.250000    0.0  0.000000    0.0   
26    0.250000    0.0  0.000000    0.0   
27    0.250000    0.0  0.000000    0.0   
28    0.250000    0.0  0.000000    0.0   
29    0.250000    0.0  0.000000    0.0   
...        ...    ...       ...    ...   
9697  0.300000    0.0  0.000000    0.0   
9698  0.033333    9.0  0.300000    2.0   
9699  0.033333    0.0  0.000000    3.0   
9700  0.066667    2.0  0.066667    3.0   
9701  0.066667    0.0  0.000000    1.0   
9702  0.033333    2.0  0.066667    0.0   
9703  0.033333   11.0  0.366667    0.0   
9704  0.066667    6.0  0.200000    0.0   
9705  0.000000    6.0  0.200000    1.0   
9706  0.133333    0.0  0.000000    5.0   
9707  0.166667    0.0  0.000000    3.0   
9708  0.100000    0.0  0.000000    2.0   
9709  0.000000    0.0  0.000000    2.0   
9710  0.100000    0.0  0.000000    8.0   
9711  0.066667    4.0  0.133333    0.0   
9712  0.200000    0.0  0.000000    0.0   
9713  0.066667    0.0  0.000000    3.0   
9714  0.100000    0.0  0.000000    1.0   
9715  0.166667    0.0  0.000000    3.0   
9716  0.000000    6.0  0.200000    2.0   
9717  0.100000    0.0  0.000000    0.0   
9718  0.000000    2.0  0.066667    1.0   
9719  0.066667    0.0  0.000000    0.0   
9720  0.066667    0.0  0.000000    0.0   
9721  0.000000    1.0  0.033333    0.0   
9722  0.033333    1.0  0.033333    0.0   
9723  0.066667    0.0  0.000000    0.0   
9724  0.466667    0.0  0.000000    7.0   
9725  0.400000    1.0  0.033333    2.0   
9726  0.400000    1.0  0.033333    2.0   

                            ...                             per_R  num_T  \
0                           ...                          0.000000    0.0   
1                           ...                          0.000000    0.0   
2                           ...                          0.000000    0.0   
3                           ...                          0.000000    1.0   
4                           ...                          0.000000    0.0   
5                           ...                          0.250000    0.0   
6                           ...                          0.250000    0.0   
7                           ...                          0.000000    0.0   
8                           ...                          0.000000    0.0   
9                           ...                          0.000000    0.0   
10                          ...                          0.000000    1.0   
11                          ...                          0.000000    0.0   
12                          ...                          0.000000    0.0   
13                          ...                          0.000000    1.0   
14                          ...                          0.000000    0.0   
15                          ...                          0.250000    0.0   
16                          ...                          0.250000    0.0   
17                          ...                          0.250000    0.0   
18                          ...                          0.000000    0.0   
19                          ...                          0.000000    0.0   
20                          ...                          0.000000    0.0   
21                          ...                          0.000000    0.0   
22                          ...                          0.250000    0.0   
23                          ...                          0.250000    0.0   
24                          ...                          0.000000    1.0   
25                          ...                          0.000000    1.0   
26                          ...                          0.000000    0.0   
27                          ...                          0.250000    0.0   
28                          ...                          0.000000    0.0   
29                          ...                          0.000000    0.0   
...                         ...                               ...    ...   
9697                        ...                          0.000000    2.0   
9698                        ...                          0.000000    3.0   
9699                        ...                          0.033333    1.0   
9700                        ...                          0.133333    0.0   
9701                        ...                          0.000000    2.0   
9702                        ...                          0.033333    2.0   
9703                        ...                          0.000000    3.0   
9704                        ...                          0.033333    1.0   
9705                        ...                          0.100000    0.0   
9706                        ...                          0.166667    3.0   
9707                        ...                          0.033333    1.0   
9708                        ...                          0.066667    2.0   
9709                        ...                          0.066667    2.0   
9710                        ...                          0.100000    1.0   
9711                        ...                          0.000000    3.0   
9712                        ...                          0.033333    1.0   
9713                        ...                          0.033333    1.0   
9714                        ...                          0.000000    1.0   
9715                        ...                          0.133333    2.0   
9716                        ...                          0.066667    1.0   
9717                        ...                          0.066667    3.0   
9718                        ...                          0.200000    1.0   
9719                        ...                          0.033333    2.0   
9720                        ...                          0.133333    0.0   
9721                        ...                          0.300000    0.0   
9722                        ...                          0.033333    0.0   
9723                        ...                          0.133333    0.0   
9724                        ...                          0.000000    0.0   
9725                        ...                          0.000000    0.0   
9726                        ...                          0.233333    0.0   

         per_T  num_W     per_W  num_V     per_V  num_Y     per_Y  \
0     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
2     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
4     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
7     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
8     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
9     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
10    0.250000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
11    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
13    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
14    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
16    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
20    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
22    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
23    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
24    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
25    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
26    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
27    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
28    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
29    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
...        ...    ...       ...    ...       ...    ...       ...   
9697  0.066667    1.0  0.033333    5.0  0.166667    2.0  0.066667   
9698  0.100000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9699  0.033333    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9700  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9701  0.066667    3.0  0.100000    1.0  0.033333    1.0  0.033333   
9702  0.066667    0.0  0.000000    0.0  0.000000    2.0  0.066667   
9703  0.100000    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9704  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9705  0.000000    0.0  0.000000    2.0  0.066667    1.0  0.033333   
9706  0.100000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9707  0.033333    1.0  0.033333    4.0  0.133333    0.0  0.000000   
9708  0.066667    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9709  0.066667    3.0  0.100000    3.0  0.100000    1.0  0.033333   
9710  0.033333    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9711  0.100000    1.0  0.033333    2.0  0.066667    2.0  0.066667   
9712  0.033333    1.0  0.033333    3.0  0.100000    0.0  0.000000   
9713  0.033333    0.0  0.000000    2.0  0.066667    0.0  0.000000   
9714  0.033333    1.0  0.033333    1.0  0.033333    2.0  0.066667   
9715  0.066667    0.0  0.000000    1.0  0.033333    4.0  0.133333   
9716  0.033333    4.0  0.133333    1.0  0.033333    1.0  0.033333   
9717  0.100000    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9718  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9719  0.066667    1.0  0.033333    1.0  0.033333    1.0  0.033333   
9720  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9721  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9722  0.000000    2.0  0.066667    4.0  0.133333    0.0  0.000000   
9723  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9724  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9725  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9726  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   

                                                      X  
0     [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...  
1     [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...  
2     [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....  
3     [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...  
4     [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...  
5     [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...  
6     [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...  
7     [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....  
8     [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...  
9     [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...  
10    [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...  
11    [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...  
12    [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...  
13    [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...  
14    [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...  
15    [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...  
16    [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...  
17    [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...  
18    [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...  
19    [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...  
20    [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...  
21    [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...  
22    [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...  
23    [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...  
24    [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...  
25    [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...  
26    [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....  
27    [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...  
28    [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...  
29    [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...  
...                                                 ...  
9697  [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...  
9698  [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...  
9699  [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...  
9700  [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....  
9701  [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...  
9702  [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...  
9703  [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...  
9704  [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...  
9705  [[14.0, 1.0, 14.0, 4.0, 15.0, 3.0, 14.0, 15.0,...  
9706  [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....  
9707  [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...  
9708  [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...  
9709  [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...  
9710  [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...  
9711  [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....  
9712  [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...  
9713  [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...  
9714  [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...  
9715  [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...  
9716  [[19.0, 1.0, 12.0, 8.0, 17.0, 9.0, 17.0, 16.0,...  
9717  [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...  
9718  [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....  
9719  [[4.0, 17.0, 16.0, 10.0, 11.0, 14.0, 0.0, 4.0,...  
9720  [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...  
9721  [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...  
9722  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...  
9723  [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...  
9724  [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...  
9725  [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...  
9726  [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...  

[9727 rows x 46 columns]
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 68, in <module>
    die
NameError: name 'die' is not defined
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> data2
                                 seq  source  size  type_aa  len  testPred
0                               SPCG      -1     4        4    4  0.007797
1                               VPSG      -1     4        4    4  0.002599
2                               ADKP       0     4        4    4  0.031334
3                               AGLT       0     4        4    4  0.002652
4                               APGW       0     4        4    4  0.001054
5                               FLRN       0     4        4    4  0.020269
6                               FYRI       0     4        4    4  0.039302
7                               GFAD       0     4        4    4  0.000825
8                               GSWD       0     4        4    4  0.003194
9                               ILME       0     4        4    4  0.041583
10                              LWKT       0     4        4    4  0.210057
11                              LWSG       0     4        4    4  0.005637
12                              LYDN       0     4        4    4  0.003981
13                              NTQM       0     4        4    4  0.019361
14                              PGLW       0     4        4    4  0.005793
15                              QGRF       0     4        4    4  0.004671
16                              RGMW       0     4        4    4  0.015456
17                              RSFN       0     4        4    4  0.012129
18                              SDKP       0     4        4    4  0.026276
19                              VGSE       0     4        4    4  0.000706
20                              WGHE       0     4        4    4  0.007200
21                              YDFI       0     4        4    4  0.004430
22                              YLRF       0     4        4    4  0.014199
23                              YMRF       0     4        4    4  0.008723
24                              AETF       1     4        4    4  0.001721
25                              AFTS       1     4        4    4  0.002515
26                              AGDV       1     4        4    4  0.001132
27                              AIRS       1     4        4    4  0.034738
28                              AKPF       1     4        4    4  0.011902
29                              ALPF       1     4        4    4  0.011916
...                              ...     ...   ...      ...  ...       ...
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.643268
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.072823
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30  0.008374
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30  0.004381
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30  0.012129
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30  0.002148
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30  0.001752
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30  0.015465
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30  0.005739
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30  0.044459
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30  0.003884
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30  0.001148
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30  0.006556
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30  0.009148
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30  0.009225
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30  0.002634
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30  0.005612
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30  0.003837
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30  0.020465
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30  0.001434
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30  0.001645
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30  0.022678
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30  0.007942
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30  0.016428
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30  0.640396
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30  0.411098
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30  0.654751
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30  0.015545
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30  0.932408
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30  0.980807

[9873 rows x 6 columns]
>>> data2
                                 seq  source  size  type_aa  len  testPred
0                               SPCG      -1     4        4    4  0.007797
1                               VPSG      -1     4        4    4  0.002599
2                               ADKP       0     4        4    4  0.031334
3                               AGLT       0     4        4    4  0.002652
4                               APGW       0     4        4    4  0.001054
5                               FLRN       0     4        4    4  0.020269
6                               FYRI       0     4        4    4  0.039302
7                               GFAD       0     4        4    4  0.000825
8                               GSWD       0     4        4    4  0.003194
9                               ILME       0     4        4    4  0.041583
10                              LWKT       0     4        4    4  0.210057
11                              LWSG       0     4        4    4  0.005637
12                              LYDN       0     4        4    4  0.003981
13                              NTQM       0     4        4    4  0.019361
14                              PGLW       0     4        4    4  0.005793
15                              QGRF       0     4        4    4  0.004671
16                              RGMW       0     4        4    4  0.015456
17                              RSFN       0     4        4    4  0.012129
18                              SDKP       0     4        4    4  0.026276
19                              VGSE       0     4        4    4  0.000706
20                              WGHE       0     4        4    4  0.007200
21                              YDFI       0     4        4    4  0.004430
22                              YLRF       0     4        4    4  0.014199
23                              YMRF       0     4        4    4  0.008723
24                              AETF       1     4        4    4  0.001721
25                              AFTS       1     4        4    4  0.002515
26                              AGDV       1     4        4    4  0.001132
27                              AIRS       1     4        4    4  0.034738
28                              AKPF       1     4        4    4  0.011902
29                              ALPF       1     4        4    4  0.011916
...                              ...     ...   ...      ...  ...       ...
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.643268
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.072823
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30  0.008374
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30  0.004381
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30  0.012129
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30  0.002148
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30  0.001752
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30  0.015465
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30  0.005739
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30  0.044459
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30  0.003884
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30  0.001148
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30  0.006556
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30  0.009148
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30  0.009225
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30  0.002634
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30  0.005612
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30  0.003837
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30  0.020465
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30  0.001434
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30  0.001645
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30  0.022678
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30  0.007942
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30  0.016428
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30  0.640396
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30  0.411098
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30  0.654751
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30  0.015545
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30  0.932408
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30  0.980807

[9873 rows x 6 columns]
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> data
                                 seq  source  size  type_aa  len
0                               SPCG      -1     4        4    4
1                               VPSG      -1     4        4    4
2                               ADKP       0     4        4    4
3                               AGLT       0     4        4    4
4                               APGW       0     4        4    4
5                               FLRN       0     4        4    4
6                               FYRI       0     4        4    4
7                               GFAD       0     4        4    4
8                               GSWD       0     4        4    4
9                               ILME       0     4        4    4
10                              LWKT       0     4        4    4
11                              LWSG       0     4        4    4
12                              LYDN       0     4        4    4
13                              NTQM       0     4        4    4
14                              PGLW       0     4        4    4
15                              QGRF       0     4        4    4
16                              RGMW       0     4        4    4
17                              RSFN       0     4        4    4
18                              SDKP       0     4        4    4
19                              VGSE       0     4        4    4
20                              WGHE       0     4        4    4
21                              YDFI       0     4        4    4
22                              YLRF       0     4        4    4
23                              YMRF       0     4        4    4
24                              AETF       1     4        4    4
25                              AFTS       1     4        4    4
26                              AGDV       1     4        4    4
27                              AIRS       1     4        4    4
28                              AKPF       1     4        4    4
29                              ALPF       1     4        4    4
...                              ...     ...   ...      ...  ...
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30

[9727 rows x 5 columns]
>>> 9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30
KeyboardInterrupt
>>> data
                                 seq  source  size  type_aa  len
0                               SPCG      -1     4        4    4
1                               VPSG      -1     4        4    4
2                               ADKP       0     4        4    4
3                               AGLT       0     4        4    4
4                               APGW       0     4        4    4
5                               FLRN       0     4        4    4
6                               FYRI       0     4        4    4
7                               GFAD       0     4        4    4
8                               GSWD       0     4        4    4
9                               ILME       0     4        4    4
10                              LWKT       0     4        4    4
11                              LWSG       0     4        4    4
12                              LYDN       0     4        4    4
13                              NTQM       0     4        4    4
14                              PGLW       0     4        4    4
15                              QGRF       0     4        4    4
16                              RGMW       0     4        4    4
17                              RSFN       0     4        4    4
18                              SDKP       0     4        4    4
19                              VGSE       0     4        4    4
20                              WGHE       0     4        4    4
21                              YDFI       0     4        4    4
22                              YLRF       0     4        4    4
23                              YMRF       0     4        4    4
24                              AETF       1     4        4    4
25                              AFTS       1     4        4    4
26                              AGDV       1     4        4    4
27                              AIRS       1     4        4    4
28                              AKPF       1     4        4    4
29                              ALPF       1     4        4    4
...                              ...     ...   ...      ...  ...
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30

[9727 rows x 5 columns]
>>> 
KeyboardInterrupt
>>> data_xgb['seq'] = data_xgb['seq'].apply(lambda x : x.upper())
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> 
KeyboardInterrupt
>>>   
KeyboardInterrupt
>>> data.merge(data_xgb[['seq','testPred']],on='seq')
                                 seq  source  size  type_aa  len  testPred
0                               SPCG      -1     4        4    4  0.007797
1                               VPSG      -1     4        4    4  0.002599
2                               ADKP       0     4        4    4  0.031334
3                               AGLT       0     4        4    4  0.002652
4                               APGW       0     4        4    4  0.001054
5                               FLRN       0     4        4    4  0.020269
6                               FYRI       0     4        4    4  0.039302
7                               GFAD       0     4        4    4  0.000825
8                               GSWD       0     4        4    4  0.003194
9                               ILME       0     4        4    4  0.041583
10                              LWKT       0     4        4    4  0.210057
11                              LWSG       0     4        4    4  0.005637
12                              LYDN       0     4        4    4  0.003981
13                              NTQM       0     4        4    4  0.019361
14                              PGLW       0     4        4    4  0.005793
15                              QGRF       0     4        4    4  0.004671
16                              RGMW       0     4        4    4  0.015456
17                              RSFN       0     4        4    4  0.012129
18                              SDKP       0     4        4    4  0.026276
19                              VGSE       0     4        4    4  0.000706
20                              WGHE       0     4        4    4  0.007200
21                              YDFI       0     4        4    4  0.004430
22                              YLRF       0     4        4    4  0.014199
23                              YMRF       0     4        4    4  0.008723
24                              AETF       1     4        4    4  0.001721
25                              AFTS       1     4        4    4  0.002515
26                              AGDV       1     4        4    4  0.001132
27                              AIRS       1     4        4    4  0.034738
28                              AKPF       1     4        4    4  0.011902
29                              ALPF       1     4        4    4  0.011916
...                              ...     ...   ...      ...  ...       ...
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.643268
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.072823
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30  0.008374
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30  0.004381
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30  0.012129
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30  0.002148
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30  0.001752
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30  0.015465
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30  0.005739
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30  0.044459
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30  0.003884
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30  0.001148
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30  0.006556
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30  0.009148
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30  0.009225
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30  0.002634
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30  0.005612
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30  0.003837
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30  0.020465
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30  0.001434
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30  0.001645
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30  0.022678
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30  0.007942
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30  0.016428
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30  0.640396
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30  0.411098
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30  0.654751
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30  0.015545
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30  0.932408
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30  0.980807

[9873 rows x 6 columns]
>>> 
KeyboardInterrupt
>>> data['seq'] = data['seq'].apply(lambda x : x.upper())
>>> data.merge(data_xgb[['seq','testPred']],on='seq')
                                 seq  source  size  type_aa  len  testPred
0                               SPCG      -1     4        4    4  0.007797
1                               VPSG      -1     4        4    4  0.002599
2                               ADKP       0     4        4    4  0.031334
3                               AGLT       0     4        4    4  0.002652
4                               APGW       0     4        4    4  0.001054
5                               FLRN       0     4        4    4  0.020269
6                               FYRI       0     4        4    4  0.039302
7                               GFAD       0     4        4    4  0.000825
8                               GSWD       0     4        4    4  0.003194
9                               ILME       0     4        4    4  0.041583
10                              LWKT       0     4        4    4  0.210057
11                              LWSG       0     4        4    4  0.005637
12                              LYDN       0     4        4    4  0.003981
13                              NTQM       0     4        4    4  0.019361
14                              PGLW       0     4        4    4  0.005793
15                              QGRF       0     4        4    4  0.004671
16                              RGMW       0     4        4    4  0.015456
17                              RSFN       0     4        4    4  0.012129
18                              SDKP       0     4        4    4  0.026276
19                              VGSE       0     4        4    4  0.000706
20                              WGHE       0     4        4    4  0.007200
21                              YDFI       0     4        4    4  0.004430
22                              YLRF       0     4        4    4  0.014199
23                              YMRF       0     4        4    4  0.008723
24                              AETF       1     4        4    4  0.001721
25                              AFTS       1     4        4    4  0.002515
26                              AGDV       1     4        4    4  0.001132
27                              AIRS       1     4        4    4  0.034738
28                              AKPF       1     4        4    4  0.011902
29                              ALPF       1     4        4    4  0.011916
...                              ...     ...   ...      ...  ...       ...
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.643268
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.072823
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30  0.008374
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30  0.004381
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30  0.012129
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30  0.002148
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30  0.001752
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30  0.015465
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30  0.005739
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30  0.044459
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30  0.003884
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30  0.001148
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30  0.006556
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30  0.009148
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30  0.009225
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30  0.002634
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30  0.005612
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30  0.003837
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30  0.020465
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30  0.001434
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30  0.001645
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30  0.022678
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30  0.007942
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30  0.016428
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30  0.640396
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30  0.411098
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30  0.654751
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30  0.015545
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30  0.932408
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30  0.980807

[9873 rows x 6 columns]
>>> data
                                 seq  source  size  type_aa  len
0                               SPCG      -1     4        4    4
1                               VPSG      -1     4        4    4
2                               ADKP       0     4        4    4
3                               AGLT       0     4        4    4
4                               APGW       0     4        4    4
5                               FLRN       0     4        4    4
6                               FYRI       0     4        4    4
7                               GFAD       0     4        4    4
8                               GSWD       0     4        4    4
9                               ILME       0     4        4    4
10                              LWKT       0     4        4    4
11                              LWSG       0     4        4    4
12                              LYDN       0     4        4    4
13                              NTQM       0     4        4    4
14                              PGLW       0     4        4    4
15                              QGRF       0     4        4    4
16                              RGMW       0     4        4    4
17                              RSFN       0     4        4    4
18                              SDKP       0     4        4    4
19                              VGSE       0     4        4    4
20                              WGHE       0     4        4    4
21                              YDFI       0     4        4    4
22                              YLRF       0     4        4    4
23                              YMRF       0     4        4    4
24                              AETF       1     4        4    4
25                              AFTS       1     4        4    4
26                              AGDV       1     4        4    4
27                              AIRS       1     4        4    4
28                              AKPF       1     4        4    4
29                              ALPF       1     4        4    4
...                              ...     ...   ...      ...  ...
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30

[9727 rows x 5 columns]
>>> data2
                                 seq  source  size  type_aa  len  testPred
0                               SPCG      -1     4        4    4  0.007797
1                               VPSG      -1     4        4    4  0.002599
2                               ADKP       0     4        4    4  0.031334
3                               AGLT       0     4        4    4  0.002652
4                               APGW       0     4        4    4  0.001054
5                               FLRN       0     4        4    4  0.020269
6                               FYRI       0     4        4    4  0.039302
7                               GFAD       0     4        4    4  0.000825
8                               GSWD       0     4        4    4  0.003194
9                               ILME       0     4        4    4  0.041583
10                              LWKT       0     4        4    4  0.210057
11                              LWSG       0     4        4    4  0.005637
12                              LYDN       0     4        4    4  0.003981
13                              NTQM       0     4        4    4  0.019361
14                              PGLW       0     4        4    4  0.005793
15                              QGRF       0     4        4    4  0.004671
16                              RGMW       0     4        4    4  0.015456
17                              RSFN       0     4        4    4  0.012129
18                              SDKP       0     4        4    4  0.026276
19                              VGSE       0     4        4    4  0.000706
20                              WGHE       0     4        4    4  0.007200
21                              YDFI       0     4        4    4  0.004430
22                              YLRF       0     4        4    4  0.014199
23                              YMRF       0     4        4    4  0.008723
24                              AETF       1     4        4    4  0.001721
25                              AFTS       1     4        4    4  0.002515
26                              AGDV       1     4        4    4  0.001132
27                              AIRS       1     4        4    4  0.034738
28                              AKPF       1     4        4    4  0.011902
29                              ALPF       1     4        4    4  0.011916
...                              ...     ...   ...      ...  ...       ...
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.643268
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30  0.072823
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30  0.008374
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30  0.004381
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30  0.012129
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30  0.002148
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30  0.001752
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30  0.015465
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30  0.005739
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30  0.044459
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30  0.003884
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30  0.001148
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30  0.006556
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30  0.009148
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30  0.009225
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30  0.002634
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30  0.005612
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30  0.003837
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30  0.020465
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30  0.001434
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30  0.001645
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30  0.022678
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30  0.007942
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30  0.016428
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30  0.640396
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30  0.411098
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30  0.654751
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30  0.015545
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30  0.932408
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30  0.980807

[9873 rows x 6 columns]
>>> data
                                 seq  source  size  type_aa  len
0                               SPCG      -1     4        4    4
1                               VPSG      -1     4        4    4
2                               ADKP       0     4        4    4
3                               AGLT       0     4        4    4
4                               APGW       0     4        4    4
5                               FLRN       0     4        4    4
6                               FYRI       0     4        4    4
7                               GFAD       0     4        4    4
8                               GSWD       0     4        4    4
9                               ILME       0     4        4    4
10                              LWKT       0     4        4    4
11                              LWSG       0     4        4    4
12                              LYDN       0     4        4    4
13                              NTQM       0     4        4    4
14                              PGLW       0     4        4    4
15                              QGRF       0     4        4    4
16                              RGMW       0     4        4    4
17                              RSFN       0     4        4    4
18                              SDKP       0     4        4    4
19                              VGSE       0     4        4    4
20                              WGHE       0     4        4    4
21                              YDFI       0     4        4    4
22                              YLRF       0     4        4    4
23                              YMRF       0     4        4    4
24                              AETF       1     4        4    4
25                              AFTS       1     4        4    4
26                              AGDV       1     4        4    4
27                              AIRS       1     4        4    4
28                              AKPF       1     4        4    4
29                              ALPF       1     4        4    4
...                              ...     ...   ...      ...  ...
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30

[9727 rows x 5 columns]
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> 
KeyboardInterrupt
>>> for i in range(len(data)):
	if data['seq'].iloc[i] != data_xgb['seq'].iloc[i]:
		print data['seq'].iloc[i] , data_xgb['seq'].iloc[i]

		
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 304, in <module>
    dropout : 0.2,learning_rate : lr}) #sgd
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
>>> data2
                            seq  source  size  type_aa  len  num_A  per_A  \
0                          SPCG      -1     4        4    4    0.0   0.00   
1                          VPSG      -1     4        4    4    0.0   0.00   
2                          ADKP       0     4        4    4    1.0   0.25   
3                          AGLT       0     4        4    4    1.0   0.25   
4                          APGW       0     4        4    4    1.0   0.25   
5                          FLRN       0     4        4    4    0.0   0.00   
6                          FYRI       0     4        4    4    0.0   0.00   
7                          GFAD       0     4        4    4    1.0   0.25   
8                          GSWD       0     4        4    4    0.0   0.00   
9                          ILME       0     4        4    4    0.0   0.00   
10                         LWKT       0     4        4    4    0.0   0.00   
11                         LWSG       0     4        4    4    0.0   0.00   
12                         LYDN       0     4        4    4    0.0   0.00   
13                         NTQM       0     4        4    4    0.0   0.00   
14                         PGLW       0     4        4    4    0.0   0.00   
15                         QGRF       0     4        4    4    0.0   0.00   
16                         RGMW       0     4        4    4    0.0   0.00   
17                         RSFN       0     4        4    4    0.0   0.00   
18                         SDKP       0     4        4    4    0.0   0.00   
19                         VGSE       0     4        4    4    0.0   0.00   
20                         WGHE       0     4        4    4    0.0   0.00   
21                         YDFI       0     4        4    4    0.0   0.00   
22                         YLRF       0     4        4    4    0.0   0.00   
23                         YMRF       0     4        4    4    0.0   0.00   
24                         AETF       1     4        4    4    1.0   0.25   
25                         AFTS       1     4        4    4    1.0   0.25   
26                         AGDV       1     4        4    4    1.0   0.25   
27                         AIRS       1     4        4    4    1.0   0.25   
28                         AKPF       1     4        4    4    1.0   0.25   
29                         ALPF       1     4        4    4    1.0   0.25   
...                         ...     ...   ...      ...  ...    ...    ...   
2459  RRDYTEQLRRAARRNAWDLYGEHFY       1    25       13   25    3.0   0.12   
2460  SETEPFFGDYCSENPDAAECLIYDD       1    25       13   25    2.0   0.08   
2461  SHMSHKKSFFDKKRSERISNCQDTS       1    25       13   25    0.0   0.00   
2462  SKCRQWQSKIRRTNPIFCIRRASPT       1    25       12   25    1.0   0.04   
2463  SKYITTIAGVMTLSQVKGFVRKNGV       1    25       14   25    1.0   0.04   
2464  SPLQAKKVRKVPPGLPSSVYAPSPN       1    25       11   25    2.0   0.08   
2465  SRRTDDEIPPPLPERTPESFIVVEE       1    25       10   25    0.0   0.00   
2466  SSSHHYSHPGGGGEQLAINELISDG       1    25       12   25    1.0   0.04   
2467  SVGTSVASAEQDELSQRLARLRDQV       1    25       10   25    3.0   0.12   
2468  SVLRTITNLQKKIRKELKQRQLKQE       1    25       10   25    0.0   0.00   
2469  SYLNGVMPPTQSFAPDPKYVSSKAL       1    25       14   25    2.0   0.08   
2470  TGNVGLSPGLSTALTGFTLVPVEDH       1    25       12   25    1.0   0.04   
2471  TNKLPVSIPLASVVLPSRAERARST       1    25       11   25    3.0   0.12   
2472  TVGDEIVDLTCESLEPVVVDLTHND       1    25       12   25    0.0   0.00   
2473  VLGKLSQELHKLQTYPRTNTGSGTP       1    25       13   25    0.0   0.00   
2474  YMLFTMIFVISSIIITVVVINTHHR       1    25       11   25    0.0   0.00   
2475  YSPTFNVAHILAFFFLFLHIPFYFV       1    25       11   25    2.0   0.08   
2476  ACSGSGSGCGSGSGSCGRRRRRRRR       2    25        5   25    1.0   0.04   
2477  ACSGSGSGCGSGSGSCGRRRRRRRR       2    25        5   25    1.0   0.04   
2478  ARRRRCSGSGSGCGSGSGSCGRRRR       2    25        5   25    1.0   0.04   
2479  ARRRRCSGSGSGCGSGSGSCGRRRR       2    25        5   25    1.0   0.04   
2480  ACSHSGHGCGHGSHSCGRRRRRRRR       2    25        6   25    1.0   0.04   
2481  ACSHSGWGCGHGSWSCGRRRRRRRR       2    25        7   25    1.0   0.04   
2482  CGGMVTVLFRRLRIRRASGPPRVRV       2    25       12   25    1.0   0.04   
2483  GTKMIFVGIKKKEERADLIAYLKKA       2    25       13   25    3.0   0.12   
2484  GWTLNSAGYLLGKLKALAALAKKIL       2    25       10   25    5.0   0.20   
2485  KHKALHALHLLALLWLHLAHLAKHK       2    25        5   25    5.0   0.20   
2486  KHKLLHLLHLLALLWLHLLHLLKHK       2    25        5   25    1.0   0.04   
2487  KKLALHALHLLALLWLHLAHLALKK       2    25        5   25    5.0   0.20   
2488  KLIKGRTPIKFGKADCDRPPKHSGK       2    25       13   25    1.0   0.04   

      num_C  per_C  num_E    ...     per_R  num_T  per_T  num_W  per_W  num_V  \
0       1.0   0.25    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
1       0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    1.0   
2       0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
3       0.0   0.00    0.0    ...      0.00    1.0   0.25    0.0   0.00    0.0   
4       0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.25    0.0   
5       0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
6       0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
7       0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
8       0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.25    0.0   
9       0.0   0.00    1.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
10      0.0   0.00    0.0    ...      0.00    1.0   0.25    1.0   0.25    0.0   
11      0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.25    0.0   
12      0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
13      0.0   0.00    0.0    ...      0.00    1.0   0.25    0.0   0.00    0.0   
14      0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.25    0.0   
15      0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
16      0.0   0.00    0.0    ...      0.25    0.0   0.00    1.0   0.25    0.0   
17      0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
18      0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
19      0.0   0.00    1.0    ...      0.00    0.0   0.00    0.0   0.00    1.0   
20      0.0   0.00    1.0    ...      0.00    0.0   0.00    1.0   0.25    0.0   
21      0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
22      0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
23      0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
24      0.0   0.00    1.0    ...      0.00    1.0   0.25    0.0   0.00    0.0   
25      0.0   0.00    0.0    ...      0.00    1.0   0.25    0.0   0.00    0.0   
26      0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    1.0   
27      0.0   0.00    0.0    ...      0.25    0.0   0.00    0.0   0.00    0.0   
28      0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
29      0.0   0.00    0.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
...     ...    ...    ...    ...       ...    ...    ...    ...    ...    ...   
2459    0.0   0.00    2.0    ...      0.24    1.0   0.04    1.0   0.04    0.0   
2460    2.0   0.08    4.0    ...      0.00    1.0   0.04    0.0   0.00    0.0   
2461    1.0   0.04    1.0    ...      0.08    1.0   0.04    0.0   0.00    0.0   
2462    2.0   0.08    0.0    ...      0.20    2.0   0.08    1.0   0.04    0.0   
2463    0.0   0.00    0.0    ...      0.04    3.0   0.12    0.0   0.00    4.0   
2464    0.0   0.00    0.0    ...      0.04    0.0   0.00    0.0   0.00    3.0   
2465    0.0   0.00    5.0    ...      0.12    2.0   0.08    0.0   0.00    2.0   
2466    0.0   0.00    2.0    ...      0.00    0.0   0.00    0.0   0.00    0.0   
2467    0.0   0.00    2.0    ...      0.12    1.0   0.04    0.0   0.00    3.0   
2468    0.0   0.00    2.0    ...      0.12    2.0   0.08    0.0   0.00    1.0   
2469    0.0   0.00    0.0    ...      0.00    1.0   0.04    0.0   0.00    2.0   
2470    0.0   0.00    1.0    ...      0.00    4.0   0.16    0.0   0.00    3.0   
2471    0.0   0.00    1.0    ...      0.12    2.0   0.08    0.0   0.00    3.0   
2472    1.0   0.04    3.0    ...      0.00    3.0   0.12    0.0   0.00    5.0   
2473    0.0   0.00    1.0    ...      0.04    4.0   0.16    0.0   0.00    1.0   
2474    0.0   0.00    0.0    ...      0.04    3.0   0.12    0.0   0.00    4.0   
2475    0.0   0.00    0.0    ...      0.00    1.0   0.04    0.0   0.00    2.0   
2476    3.0   0.12    0.0    ...      0.32    0.0   0.00    0.0   0.00    0.0   
2477    3.0   0.12    0.0    ...      0.32    0.0   0.00    0.0   0.00    0.0   
2478    3.0   0.12    0.0    ...      0.32    0.0   0.00    0.0   0.00    0.0   
2479    3.0   0.12    0.0    ...      0.32    0.0   0.00    0.0   0.00    0.0   
2480    3.0   0.12    0.0    ...      0.32    0.0   0.00    0.0   0.00    0.0   
2481    3.0   0.12    0.0    ...      0.32    0.0   0.00    2.0   0.08    0.0   
2482    1.0   0.04    0.0    ...      0.28    1.0   0.04    0.0   0.00    4.0   
2483    0.0   0.00    2.0    ...      0.04    1.0   0.04    0.0   0.00    1.0   
2484    0.0   0.00    0.0    ...      0.00    1.0   0.04    1.0   0.04    0.0   
2485    0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.04    0.0   
2486    0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.04    0.0   
2487    0.0   0.00    0.0    ...      0.00    0.0   0.00    1.0   0.04    0.0   
2488    1.0   0.04    0.0    ...      0.08    1.0   0.04    0.0   0.00    0.0   

      per_V  num_Y  per_Y  testPred  
0      0.00    0.0   0.00  0.007797  
1      0.25    0.0   0.00  0.002599  
2      0.00    0.0   0.00  0.031334  
3      0.00    0.0   0.00  0.002652  
4      0.00    0.0   0.00  0.001054  
5      0.00    0.0   0.00  0.020269  
6      0.00    1.0   0.25  0.039302  
7      0.00    0.0   0.00  0.000825  
8      0.00    0.0   0.00  0.003194  
9      0.00    0.0   0.00  0.041583  
10     0.00    0.0   0.00  0.210057  
11     0.00    0.0   0.00  0.005637  
12     0.00    1.0   0.25  0.003981  
13     0.00    0.0   0.00  0.019361  
14     0.00    0.0   0.00  0.005793  
15     0.00    0.0   0.00  0.004671  
16     0.00    0.0   0.00  0.015456  
17     0.00    0.0   0.00  0.012129  
18     0.00    0.0   0.00  0.026276  
19     0.25    0.0   0.00  0.000706  
20     0.00    0.0   0.00  0.007200  
21     0.00    1.0   0.25  0.004430  
22     0.00    1.0   0.25  0.014199  
23     0.00    1.0   0.25  0.008723  
24     0.00    0.0   0.00  0.001721  
25     0.00    0.0   0.00  0.002515  
26     0.25    0.0   0.00  0.001132  
27     0.00    0.0   0.00  0.034738  
28     0.00    0.0   0.00  0.011902  
29     0.00    0.0   0.00  0.011916  
...     ...    ...    ...       ...  
2459   0.00    3.0   0.12  0.019816  
2460   0.00    2.0   0.08  0.001201  
2461   0.00    0.0   0.00  0.038845  
2462   0.00    0.0   0.00  0.146400  
2463   0.16    1.0   0.04  0.008439  
2464   0.12    1.0   0.04  0.014037  
2465   0.08    0.0   0.00  0.058019  
2466   0.00    1.0   0.04  0.003134  
2467   0.12    0.0   0.00  0.000870  
2468   0.04    0.0   0.00  0.033332  
2469   0.08    2.0   0.08  0.002923  
2470   0.12    0.0   0.00  0.004368  
2471   0.12    0.0   0.00  0.003399  
2472   0.20    0.0   0.00  0.021303  
2473   0.04    1.0   0.04  0.014238  
2474   0.16    1.0   0.04  0.035456  
2475   0.08    2.0   0.08  0.008402  
2476   0.00    0.0   0.00  0.960407  
2477   0.00    0.0   0.00  0.972781  
2478   0.00    0.0   0.00  0.960407  
2479   0.00    0.0   0.00  0.972781  
2480   0.00    0.0   0.00  0.919023  
2481   0.00    0.0   0.00  0.868070  
2482   0.16    0.0   0.00  0.377169  
2483   0.04    1.0   0.04  0.010001  
2484   0.00    1.0   0.04  0.666969  
2485   0.00    0.0   0.00  0.986693  
2486   0.00    0.0   0.00  0.902972  
2487   0.00    0.0   0.00  0.984180  
2488   0.00    0.0   0.00  0.638177  

[2489 rows x 46 columns]
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[['testPred',]+features[::2]],on=features)

Traceback (most recent call last):
  File "<pyshell#64>", line 1, in <module>
    data.merge(data_xgb[['testPred',]+features[::2]],on=features)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1976, in _getitem_column
    result = result[key]
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'num_A'
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[['testPred',]+features],on=features[::2])

Traceback (most recent call last):
  File "<pyshell#65>", line 1, in <module>
    data.merge(data_xgb[['testPred',]+features],on=features[::2])
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 823, in _get_merge_keys
    left_keys.append(left[lk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'testPred'
>>> data.merge(data_xgb[['testPred',]+features],on=features[::2])

Traceback (most recent call last):
  File "<pyshell#66>", line 1, in <module>
    data.merge(data_xgb[['testPred',]+features],on=features[::2])
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 823, in _get_merge_keys
    left_keys.append(left[lk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'testPred'
>>> data
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E  \
0     0.000000    1.0  0.250000    0.0   
1     0.000000    0.0  0.000000    0.0   
2     0.250000    0.0  0.000000    0.0   
3     0.250000    0.0  0.000000    0.0   
4     0.250000    0.0  0.000000    0.0   
5     0.000000    0.0  0.000000    0.0   
6     0.000000    0.0  0.000000    0.0   
7     0.250000    0.0  0.000000    0.0   
8     0.000000    0.0  0.000000    0.0   
9     0.000000    0.0  0.000000    1.0   
10    0.000000    0.0  0.000000    0.0   
11    0.000000    0.0  0.000000    0.0   
12    0.000000    0.0  0.000000    0.0   
13    0.000000    0.0  0.000000    0.0   
14    0.000000    0.0  0.000000    0.0   
15    0.000000    0.0  0.000000    0.0   
16    0.000000    0.0  0.000000    0.0   
17    0.000000    0.0  0.000000    0.0   
18    0.000000    0.0  0.000000    0.0   
19    0.000000    0.0  0.000000    1.0   
20    0.000000    0.0  0.000000    1.0   
21    0.000000    0.0  0.000000    0.0   
22    0.000000    0.0  0.000000    0.0   
23    0.000000    0.0  0.000000    0.0   
24    0.250000    0.0  0.000000    1.0   
25    0.250000    0.0  0.000000    0.0   
26    0.250000    0.0  0.000000    0.0   
27    0.250000    0.0  0.000000    0.0   
28    0.250000    0.0  0.000000    0.0   
29    0.250000    0.0  0.000000    0.0   
...        ...    ...       ...    ...   
9697  0.300000    0.0  0.000000    0.0   
9698  0.033333    9.0  0.300000    2.0   
9699  0.033333    0.0  0.000000    3.0   
9700  0.066667    2.0  0.066667    3.0   
9701  0.066667    0.0  0.000000    1.0   
9702  0.033333    2.0  0.066667    0.0   
9703  0.033333   11.0  0.366667    0.0   
9704  0.066667    6.0  0.200000    0.0   
9705  0.000000    6.0  0.200000    1.0   
9706  0.133333    0.0  0.000000    5.0   
9707  0.166667    0.0  0.000000    3.0   
9708  0.100000    0.0  0.000000    2.0   
9709  0.000000    0.0  0.000000    2.0   
9710  0.100000    0.0  0.000000    8.0   
9711  0.066667    4.0  0.133333    0.0   
9712  0.200000    0.0  0.000000    0.0   
9713  0.066667    0.0  0.000000    3.0   
9714  0.100000    0.0  0.000000    1.0   
9715  0.166667    0.0  0.000000    3.0   
9716  0.000000    6.0  0.200000    2.0   
9717  0.100000    0.0  0.000000    0.0   
9718  0.000000    2.0  0.066667    1.0   
9719  0.066667    0.0  0.000000    0.0   
9720  0.066667    0.0  0.000000    0.0   
9721  0.000000    1.0  0.033333    0.0   
9722  0.033333    1.0  0.033333    0.0   
9723  0.066667    0.0  0.000000    0.0   
9724  0.466667    0.0  0.000000    7.0   
9725  0.400000    1.0  0.033333    2.0   
9726  0.400000    1.0  0.033333    2.0   

                            ...                             per_R  num_T  \
0                           ...                          0.000000    0.0   
1                           ...                          0.000000    0.0   
2                           ...                          0.000000    0.0   
3                           ...                          0.000000    1.0   
4                           ...                          0.000000    0.0   
5                           ...                          0.250000    0.0   
6                           ...                          0.250000    0.0   
7                           ...                          0.000000    0.0   
8                           ...                          0.000000    0.0   
9                           ...                          0.000000    0.0   
10                          ...                          0.000000    1.0   
11                          ...                          0.000000    0.0   
12                          ...                          0.000000    0.0   
13                          ...                          0.000000    1.0   
14                          ...                          0.000000    0.0   
15                          ...                          0.250000    0.0   
16                          ...                          0.250000    0.0   
17                          ...                          0.250000    0.0   
18                          ...                          0.000000    0.0   
19                          ...                          0.000000    0.0   
20                          ...                          0.000000    0.0   
21                          ...                          0.000000    0.0   
22                          ...                          0.250000    0.0   
23                          ...                          0.250000    0.0   
24                          ...                          0.000000    1.0   
25                          ...                          0.000000    1.0   
26                          ...                          0.000000    0.0   
27                          ...                          0.250000    0.0   
28                          ...                          0.000000    0.0   
29                          ...                          0.000000    0.0   
...                         ...                               ...    ...   
9697                        ...                          0.000000    2.0   
9698                        ...                          0.000000    3.0   
9699                        ...                          0.033333    1.0   
9700                        ...                          0.133333    0.0   
9701                        ...                          0.000000    2.0   
9702                        ...                          0.033333    2.0   
9703                        ...                          0.000000    3.0   
9704                        ...                          0.033333    1.0   
9705                        ...                          0.100000    0.0   
9706                        ...                          0.166667    3.0   
9707                        ...                          0.033333    1.0   
9708                        ...                          0.066667    2.0   
9709                        ...                          0.066667    2.0   
9710                        ...                          0.100000    1.0   
9711                        ...                          0.000000    3.0   
9712                        ...                          0.033333    1.0   
9713                        ...                          0.033333    1.0   
9714                        ...                          0.000000    1.0   
9715                        ...                          0.133333    2.0   
9716                        ...                          0.066667    1.0   
9717                        ...                          0.066667    3.0   
9718                        ...                          0.200000    1.0   
9719                        ...                          0.033333    2.0   
9720                        ...                          0.133333    0.0   
9721                        ...                          0.300000    0.0   
9722                        ...                          0.033333    0.0   
9723                        ...                          0.133333    0.0   
9724                        ...                          0.000000    0.0   
9725                        ...                          0.000000    0.0   
9726                        ...                          0.233333    0.0   

         per_T  num_W     per_W  num_V     per_V  num_Y     per_Y  \
0     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
2     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
4     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
7     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
8     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
9     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
10    0.250000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
11    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
13    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
14    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
16    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
20    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
22    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
23    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000   
24    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
25    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
26    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000   
27    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
28    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
29    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
...        ...    ...       ...    ...       ...    ...       ...   
9697  0.066667    1.0  0.033333    5.0  0.166667    2.0  0.066667   
9698  0.100000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9699  0.033333    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9700  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9701  0.066667    3.0  0.100000    1.0  0.033333    1.0  0.033333   
9702  0.066667    0.0  0.000000    0.0  0.000000    2.0  0.066667   
9703  0.100000    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9704  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9705  0.000000    0.0  0.000000    2.0  0.066667    1.0  0.033333   
9706  0.100000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9707  0.033333    1.0  0.033333    4.0  0.133333    0.0  0.000000   
9708  0.066667    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9709  0.066667    3.0  0.100000    3.0  0.100000    1.0  0.033333   
9710  0.033333    0.0  0.000000    1.0  0.033333    1.0  0.033333   
9711  0.100000    1.0  0.033333    2.0  0.066667    2.0  0.066667   
9712  0.033333    1.0  0.033333    3.0  0.100000    0.0  0.000000   
9713  0.033333    0.0  0.000000    2.0  0.066667    0.0  0.000000   
9714  0.033333    1.0  0.033333    1.0  0.033333    2.0  0.066667   
9715  0.066667    0.0  0.000000    1.0  0.033333    4.0  0.133333   
9716  0.033333    4.0  0.133333    1.0  0.033333    1.0  0.033333   
9717  0.100000    0.0  0.000000    3.0  0.100000    2.0  0.066667   
9718  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000   
9719  0.066667    1.0  0.033333    1.0  0.033333    1.0  0.033333   
9720  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9721  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333   
9722  0.000000    2.0  0.066667    4.0  0.133333    0.0  0.000000   
9723  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000   
9724  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9725  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   
9726  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000   

                                                      X  
0     [[14.0, 13.0, 1.0, 4.0], [[0.13, 0.45, -0.24, ...  
1     [[18.0, 13.0, 14.0, 4.0], [[0.07, 0.45, 0.13, ...  
2     [[0.0, 3.0, 8.0, 13.0], [[0.17, 1.23, 0.99, 0....  
3     [[0.0, 4.0, 10.0, 16.0], [[0.17, 0.01, -0.56, ...  
4     [[0.0, 13.0, 4.0, 17.0], [[0.17, 0.45, 0.01, -...  
5     [[5.0, 10.0, 15.0, 11.0], [[-1.13, -0.56, 0.81...  
6     [[5.0, 19.0, 15.0, 6.0], [[-1.13, -0.94, 0.81,...  
7     [[4.0, 5.0, 0.0, 3.0], [[0.01, -1.13, 0.17, 1....  
8     [[4.0, 14.0, 17.0, 3.0], [[0.01, 0.13, -1.85, ...  
9     [[6.0, 10.0, 9.0, 2.0], [[-0.31, -0.56, -0.23,...  
10    [[10.0, 17.0, 8.0, 16.0], [[-0.56, -1.85, 0.99...  
11    [[10.0, 17.0, 14.0, 4.0], [[-0.56, -1.85, 0.13...  
12    [[10.0, 19.0, 3.0, 11.0], [[-0.56, -0.94, 1.23...  
13    [[11.0, 16.0, 12.0, 9.0], [[0.42, 0.14, 0.58, ...  
14    [[13.0, 4.0, 10.0, 17.0], [[0.45, 0.01, -0.56,...  
15    [[12.0, 4.0, 15.0, 5.0], [[0.58, 0.01, 0.81, -...  
16    [[15.0, 4.0, 9.0, 17.0], [[0.81, 0.01, -0.23, ...  
17    [[15.0, 14.0, 5.0, 11.0], [[0.81, 0.13, -1.13,...  
18    [[14.0, 3.0, 8.0, 13.0], [[0.13, 1.23, 0.99, 0...  
19    [[18.0, 4.0, 14.0, 2.0], [[0.07, 0.01, 0.13, 2...  
20    [[17.0, 4.0, 7.0, 2.0], [[-1.85, 0.01, 0.96, 2...  
21    [[19.0, 3.0, 5.0, 6.0], [[-0.94, 1.23, -1.13, ...  
22    [[19.0, 10.0, 15.0, 5.0], [[-0.94, -0.56, 0.81...  
23    [[19.0, 9.0, 15.0, 5.0], [[-0.94, -0.23, 0.81,...  
24    [[0.0, 2.0, 16.0, 5.0], [[0.17, 2.02, 0.14, -1...  
25    [[0.0, 5.0, 16.0, 14.0], [[0.17, -1.13, 0.14, ...  
26    [[0.0, 4.0, 3.0, 18.0], [[0.17, 0.01, 1.23, 0....  
27    [[0.0, 6.0, 15.0, 14.0], [[0.17, -0.31, 0.81, ...  
28    [[0.0, 8.0, 13.0, 5.0], [[0.17, 0.99, 0.45, -1...  
29    [[0.0, 10.0, 13.0, 5.0], [[0.17, -0.56, 0.45, ...  
...                                                 ...  
9697  [[13.0, 0.0, 6.0, 19.0, 6.0, 4.0, 0.0, 16.0, 1...  
9698  [[13.0, 1.0, 2.0, 8.0, 1.0, 16.0, 14.0, 4.0, 1...  
9699  [[13.0, 5.0, 13.0, 13.0, 16.0, 13.0, 13.0, 4.0...  
9700  [[12.0, 12.0, 0.0, 15.0, 12.0, 11.0, 10.0, 12....  
9701  [[12.0, 16.0, 11.0, 17.0, 12.0, 8.0, 10.0, 2.0...  
9702  [[15.0, 13.0, 19.0, 7.0, 1.0, 14.0, 19.0, 1.0,...  
9703  [[14.0, 1.0, 1.0, 13.0, 1.0, 1.0, 13.0, 14.0, ...  
9704  [[14.0, 1.0, 11.0, 11.0, 14.0, 1.0, 12.0, 14.0...  
9705  [[14.0, 1.0, 14.0, 4.0, 15.0, 3.0, 14.0, 15.0,...  
9706  [[14.0, 2.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0, 19....  
9707  [[16.0, 0.0, 5.0, 18.0, 2.0, 13.0, 5.0, 18.0, ...  
9708  [[16.0, 13.0, 18.0, 2.0, 15.0, 12.0, 16.0, 6.0...  
9709  [[16.0, 18.0, 5.0, 16.0, 14.0, 17.0, 2.0, 2.0,...  
9710  [[18.0, 2.0, 8.0, 10.0, 16.0, 0.0, 3.0, 0.0, 2...  
9711  [[18.0, 6.0, 7.0, 1.0, 3.0, 0.0, 0.0, 16.0, 6....  
9712  [[18.0, 8.0, 4.0, 15.0, 6.0, 3.0, 0.0, 13.0, 3...  
9713  [[18.0, 16.0, 18.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3...  
9714  [[19.0, 0.0, 2.0, 4.0, 16.0, 5.0, 6.0, 14.0, 3...  
9715  [[19.0, 0.0, 2.0, 8.0, 18.0, 0.0, 12.0, 2.0, 8...  
9716  [[19.0, 1.0, 12.0, 8.0, 17.0, 9.0, 17.0, 16.0,...  
9717  [[0.0, 3.0, 18.0, 5.0, 3.0, 15.0, 4.0, 4.0, 13...  
9718  [[1.0, 4.0, 4.0, 8.0, 3.0, 1.0, 2.0, 15.0, 15....  
9719  [[4.0, 17.0, 16.0, 10.0, 11.0, 14.0, 0.0, 4.0,...  
9720  [[10.0, 10.0, 6.0, 6.0, 10.0, 15.0, 15.0, 15.0...  
9721  [[9.0, 15.0, 15.0, 6.0, 15.0, 13.0, 15.0, 13.0...  
9722  [[9.0, 18.0, 8.0, 14.0, 8.0, 6.0, 4.0, 14.0, 1...  
9723  [[11.0, 7.0, 12.0, 12.0, 12.0, 11.0, 13.0, 7.0...  
9724  [[17.0, 2.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 10...  
9725  [[17.0, 2.0, 0.0, 8.0, 10.0, 0.0, 8.0, 0.0, 10...  
9726  [[17.0, 2.0, 0.0, 15.0, 10.0, 0.0, 15.0, 0.0, ...  

[9727 rows x 46 columns]
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[['testPred',]+features],on=features[::2])

Traceback (most recent call last):
  File "<pyshell#68>", line 1, in <module>
    data.merge(data_xgb[['testPred',]+features],on=features[::2])
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 823, in _get_merge_keys
    left_keys.append(left[lk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'testPred'
>>> data_xgb
                                 seq  source  size  type_aa  len  \
0                               SPCG      -1     4        4    4   
1                               VPSG      -1     4        4    4   
2                               ADKP       0     4        4    4   
3                               AGLT       0     4        4    4   
4                               APGW       0     4        4    4   
5                               FLRN       0     4        4    4   
6                               FYRI       0     4        4    4   
7                               GFAD       0     4        4    4   
8                               GSWD       0     4        4    4   
9                               ILME       0     4        4    4   
10                              LWKT       0     4        4    4   
11                              LWSG       0     4        4    4   
12                              LYDN       0     4        4    4   
13                              NTQM       0     4        4    4   
14                              PGLW       0     4        4    4   
15                              QGRF       0     4        4    4   
16                              RGMW       0     4        4    4   
17                              RSFN       0     4        4    4   
18                              SDKP       0     4        4    4   
19                              VGSE       0     4        4    4   
20                              WGHE       0     4        4    4   
21                              YDFI       0     4        4    4   
22                              YLRF       0     4        4    4   
23                              YMRF       0     4        4    4   
24                              AETF       1     4        4    4   
25                              AFTS       1     4        4    4   
26                              AGDV       1     4        4    4   
27                              AIRS       1     4        4    4   
28                              AKPF       1     4        4    4   
29                              ALPF       1     4        4    4   
...                              ...     ...   ...      ...  ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   

                                                      X  num_A     per_A  \
0     [array([14.000, 13.000, 1.000, 4.000]), array(...    0.0  0.000000   
1     [array([18.000, 13.000, 14.000, 4.000]), array...    0.0  0.000000   
2     [array([0.000, 3.000, 8.000, 13.000]), array([...    1.0  0.250000   
3     [array([0.000, 4.000, 10.000, 16.000]), array(...    1.0  0.250000   
4     [array([0.000, 13.000, 4.000, 17.000]), array(...    1.0  0.250000   
5     [array([5.000, 10.000, 15.000, 11.000]), array...    0.0  0.000000   
6     [array([5.000, 19.000, 15.000, 6.000]), array(...    0.0  0.000000   
7     [array([4.000, 5.000, 0.000, 3.000]), array([[...    1.0  0.250000   
8     [array([4.000, 14.000, 17.000, 3.000]), array(...    0.0  0.000000   
9     [array([6.000, 10.000, 9.000, 2.000]), array([...    0.0  0.000000   
10    [array([10.000, 17.000, 8.000, 16.000]), array...    0.0  0.000000   
11    [array([10.000, 17.000, 14.000, 4.000]), array...    0.0  0.000000   
12    [array([10.000, 19.000, 3.000, 11.000]), array...    0.0  0.000000   
13    [array([11.000, 16.000, 12.000, 9.000]), array...    0.0  0.000000   
14    [array([13.000, 4.000, 10.000, 17.000]), array...    0.0  0.000000   
15    [array([12.000, 4.000, 15.000, 5.000]), array(...    0.0  0.000000   
16    [array([15.000, 4.000, 9.000, 17.000]), array(...    0.0  0.000000   
17    [array([15.000, 14.000, 5.000, 11.000]), array...    0.0  0.000000   
18    [array([14.000, 3.000, 8.000, 13.000]), array(...    0.0  0.000000   
19    [array([18.000, 4.000, 14.000, 2.000]), array(...    0.0  0.000000   
20    [array([17.000, 4.000, 7.000, 2.000]), array([...    0.0  0.000000   
21    [array([19.000, 3.000, 5.000, 6.000]), array([...    0.0  0.000000   
22    [array([19.000, 10.000, 15.000, 5.000]), array...    0.0  0.000000   
23    [array([19.000, 9.000, 15.000, 5.000]), array(...    0.0  0.000000   
24    [array([0.000, 2.000, 16.000, 5.000]), array([...    1.0  0.250000   
25    [array([0.000, 5.000, 16.000, 14.000]), array(...    1.0  0.250000   
26    [array([0.000, 4.000, 3.000, 18.000]), array([...    1.0  0.250000   
27    [array([0.000, 6.000, 15.000, 14.000]), array(...    1.0  0.250000   
28    [array([0.000, 8.000, 13.000, 5.000]), array([...    1.0  0.250000   
29    [array([0.000, 10.000, 13.000, 5.000]), array(...    1.0  0.250000   
...                                                 ...    ...       ...   
9697  [array([13.000, 0.000, 6.000, 19.000, 6.000, 4...    9.0  0.300000   
9698  [array([13.000, 1.000, 2.000, 8.000, 1.000, 16...    1.0  0.033333   
9699  [array([13.000, 5.000, 13.000, 13.000, 16.000,...    1.0  0.033333   
9700  [array([12.000, 12.000, 0.000, 15.000, 12.000,...    2.0  0.066667   
9701  [array([12.000, 16.000, 11.000, 17.000, 12.000...    2.0  0.066667   
9702  [array([15.000, 13.000, 19.000, 7.000, 1.000, ...    1.0  0.033333   
9703  [array([14.000, 1.000, 1.000, 13.000, 1.000, 1...    1.0  0.033333   
9704  [array([14.000, 1.000, 11.000, 11.000, 14.000,...    2.0  0.066667   
9705  [array([14.000, 1.000, 14.000, 4.000, 15.000, ...    0.0  0.000000   
9706  [array([14.000, 2.000, 8.000, 0.000, 0.000, 2....    4.0  0.133333   
9707  [array([16.000, 0.000, 5.000, 18.000, 2.000, 1...    5.0  0.166667   
9708  [array([16.000, 13.000, 18.000, 2.000, 15.000,...    3.0  0.100000   
9709  [array([16.000, 18.000, 5.000, 16.000, 14.000,...    0.0  0.000000   
9710  [array([18.000, 2.000, 8.000, 10.000, 16.000, ...    3.0  0.100000   
9711  [array([18.000, 6.000, 7.000, 1.000, 3.000, 0....    2.0  0.066667   
9712  [array([18.000, 8.000, 4.000, 15.000, 6.000, 3...    6.0  0.200000   
9713  [array([18.000, 16.000, 18.000, 3.000, 3.000, ...    2.0  0.066667   
9714  [array([19.000, 0.000, 2.000, 4.000, 16.000, 5...    3.0  0.100000   
9715  [array([19.000, 0.000, 2.000, 8.000, 18.000, 0...    5.0  0.166667   
9716  [array([19.000, 1.000, 12.000, 8.000, 17.000, ...    0.0  0.000000   
9717  [array([0.000, 3.000, 18.000, 5.000, 3.000, 15...    3.0  0.100000   
9718  [array([1.000, 4.000, 4.000, 8.000, 3.000, 1.0...    0.0  0.000000   
9719  [array([4.000, 17.000, 16.000, 10.000, 11.000,...    2.0  0.066667   
9720  [array([10.000, 10.000, 6.000, 6.000, 10.000, ...    2.0  0.066667   
9721  [array([9.000, 15.000, 15.000, 6.000, 15.000, ...    0.0  0.000000   
9722  [array([9.000, 18.000, 8.000, 14.000, 8.000, 6...    1.0  0.033333   
9723  [array([11.000, 7.000, 12.000, 12.000, 12.000,...    2.0  0.066667   
9724  [array([17.000, 2.000, 0.000, 0.000, 10.000, 0...   14.0  0.466667   
9725  [array([17.000, 2.000, 0.000, 8.000, 10.000, 0...   12.0  0.400000   
9726  [array([17.000, 2.000, 0.000, 15.000, 10.000, ...   12.0  0.400000   

      num_C     per_C  ...       per_V  num_Y     per_Y  length  netcharge  \
0       1.0  0.250000  ...    0.000000    0.0  0.000000       4        0.0   
1       0.0  0.000000  ...    0.250000    0.0  0.000000       4        0.0   
2       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
3       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
4       0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
5       0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
6       0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
7       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
8       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
9       0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
10      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
11      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
12      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
13      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
14      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
15      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
16      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
17      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
18      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
19      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
20      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
21      0.0  0.000000  ...    0.000000    1.0  0.250000       4       -1.0   
22      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
23      0.0  0.000000  ...    0.000000    1.0  0.250000       4        1.0   
24      0.0  0.000000  ...    0.000000    0.0  0.000000       4       -1.0   
25      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
26      0.0  0.000000  ...    0.250000    0.0  0.000000       4       -1.0   
27      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
28      0.0  0.000000  ...    0.000000    0.0  0.000000       4        1.0   
29      0.0  0.000000  ...    0.000000    0.0  0.000000       4        0.0   
...     ...       ...  ...         ...    ...       ...     ...        ...   
9697    0.0  0.000000  ...    0.166667    2.0  0.066667      30        0.0   
9698    9.0  0.300000  ...    0.000000    0.0  0.000000      30        2.0   
9699    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -4.0   
9700    2.0  0.066667  ...    0.000000    1.0  0.033333      30        2.0   
9701    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9702    2.0  0.066667  ...    0.000000    2.0  0.066667      30        8.0   
9703   11.0  0.366667  ...    0.033333    0.0  0.000000      30        2.0   
9704    6.0  0.200000  ...    0.033333    0.0  0.000000      30        0.0   
9705    6.0  0.200000  ...    0.066667    1.0  0.033333      30        2.0   
9706    0.0  0.000000  ...    0.000000    1.0  0.033333      30       -2.0   
9707    0.0  0.000000  ...    0.133333    0.0  0.000000      30       -2.0   
9708    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9709    0.0  0.000000  ...    0.100000    1.0  0.033333      30       -2.0   
9710    0.0  0.000000  ...    0.033333    1.0  0.033333      30       -4.0   
9711    4.0  0.133333  ...    0.066667    2.0  0.066667      30       -2.0   
9712    0.0  0.000000  ...    0.100000    0.0  0.000000      30        1.0   
9713    0.0  0.000000  ...    0.066667    0.0  0.000000      30       -6.0   
9714    0.0  0.000000  ...    0.033333    2.0  0.066667      30       -2.0   
9715    0.0  0.000000  ...    0.033333    4.0  0.133333      30        3.0   
9716    6.0  0.200000  ...    0.033333    1.0  0.033333      30        4.0   
9717    0.0  0.000000  ...    0.100000    2.0  0.066667      30       -2.0   
9718    2.0  0.066667  ...    0.033333    0.0  0.000000      30        6.0   
9719    0.0  0.000000  ...    0.033333    1.0  0.033333      30        1.0   
9720    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9721    1.0  0.033333  ...    0.000000    1.0  0.033333      30        9.0   
9722    1.0  0.033333  ...    0.133333    0.0  0.000000      30        5.0   
9723    0.0  0.000000  ...    0.000000    0.0  0.000000      30        6.0   
9724    0.0  0.000000  ...    0.000000    0.0  0.000000      30       -7.0   
9725    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   
9726    1.0  0.033333  ...    0.000000    0.0  0.000000      30        5.0   

       Gwif   Goct  y  testPred  testY  
0      1.38   1.73  0  0.007797    0.0  
1      0.63   1.29  0  0.002599    0.0  
2      4.24   7.08  0  0.031334    0.0  
3      0.89   0.65  0  0.002652    0.0  
4      0.92  -0.30  0  0.001054    0.0  
5      0.16  -0.30  0  0.020269    0.0  
6     -0.16  -1.73  0  0.039302    0.0  
7      3.30   3.58  0  0.000825    0.0  
8      3.64   3.16  0  0.003194    0.0  
9     -0.33   0.59  0  0.041583    0.0  
10     0.99  -0.29  0  0.210057    0.0  
11     0.54  -1.73  0  0.005637    0.0  
12     2.38   2.53  0  0.003981    0.0  
13     0.29   1.20  0  0.019361    0.0  
14    -0.10  -2.05  0  0.005793    0.0  
15     1.75   2.02  0  0.004671    0.0  
16     1.46   0.20  0  0.015456    0.0  
17     1.18   1.41  0  0.012129    0.0  
18     4.24   7.04  0  0.026276    0.0  
19     2.55   4.78  0  0.000706    0.0  
20     3.88   5.02  0  0.007200    0.0  
21     1.25   0.10  0  0.004430    0.0  
22    -0.04  -1.86  0  0.014199    0.0  
23     0.21  -1.28  0  0.008723    0.0  
24     1.47   2.67  0  0.001721    0.0  
25     0.19  -0.50  0  0.002515    0.0  
26     3.35   4.83  0  0.001132    0.0  
27     0.85   1.65  0  0.034738    0.0  
28     1.25   1.73  0  0.011902    0.0  
29    -1.25  -2.32  0  0.011916    0.0  
...     ...    ... ..       ...    ...  
9697   1.32  -0.51  0  0.012129    0.0  
9698  18.54  29.52  0  0.002148    0.0  
9699   6.44  14.87  0  0.001752    0.0  
9700   7.89  17.30  0  0.015465    0.0  
9701   5.54   2.77  0  0.005739    0.0  
9702  19.95  27.43  0  0.044459    0.0  
9703  14.70  18.95  0  0.003884    0.0  
9704  11.87  15.91  0  0.001148    0.0  
9705  12.53  17.22  0  0.000774    0.0  
9706  23.01  40.47  0  0.006556    0.0  
9707   1.07   4.48  0  0.009148    0.0  
9708   5.22  15.00  0  0.009225    0.0  
9709   7.10   5.91  0  0.002634    0.0  
9710  20.89  41.20  0  0.005612    0.0  
9711   7.00   6.21  0  0.003837    0.0  
9712  13.37  19.81  0  0.020465    0.0  
9713  25.02  43.40  0  0.001434    0.0  
9714  12.01  15.54  0  0.001645    0.0  
9715  14.62  20.51  0  0.022678    0.0  
9716  16.25  17.66  0  0.013341    0.0  
9717  11.50  16.02  1  0.007942    1.0  
9718  23.11  36.73  1  0.016428    1.0  
9719  12.96  14.72  1  0.003055    1.0  
9720  10.03  21.81  1  0.640396    1.0  
9721   6.25  14.55  1  0.411098    1.0  
9722   7.91   8.83  1  0.072823    1.0  
9723  10.03  21.81  1  0.654751    1.0  
9724  12.19  23.90  1  0.015545    1.0  
9725  17.06  25.58  1  0.932408    1.0  
9726  11.39  18.65  1  0.980807    1.0  

[9727 rows x 53 columns]
>>> 
KeyboardInterrupt
>>> features[::2]
['per_A', 'per_C', 'per_E', 'per_D', 'per_G', 'per_F', 'per_I', 'per_H', 'per_K', 'per_M', 'per_L', 'per_N', 'per_Q', 'per_P', 'per_S', 'per_R', 'per_T', 'per_W', 'per_V', 'per_Y', 'testPred']
>>> ['per_A', 'per_C', 'per_E', 'per_D', 'per_G', 'per_F', 'per_I', 'per_H', 'per_K', 'per_M', 'per_L', 'per_N', 'per_Q', 'per_P', 'per_S', 'per_R', 'per_T', 'per_W', 'per_V', 'per_Y', 'testPred']
KeyboardInterrupt
>>> features[1::2]
['num_A', 'num_C', 'num_E', 'num_D', 'num_G', 'num_F', 'num_I', 'num_H', 'num_K', 'num_M', 'num_L', 'num_N', 'num_Q', 'num_P', 'num_S', 'num_R', 'num_T', 'num_W', 'num_V', 'num_Y']
>>> data.merge(data_xgb[['testPred',]+features],on=features[1::2])
                                  seq  source  size  type_aa  len  num_A  \
0                                SPCG      -1     4        4    4    0.0   
1                                VPSG      -1     4        4    4    0.0   
2                                ADKP       0     4        4    4    1.0   
3                                AGLT       0     4        4    4    1.0   
4                                APGW       0     4        4    4    1.0   
5                                FLRN       0     4        4    4    0.0   
6                                FYRI       0     4        4    4    0.0   
7                                GFAD       0     4        4    4    1.0   
8                                GSWD       0     4        4    4    0.0   
9                                ILME       0     4        4    4    0.0   
10                               LWKT       0     4        4    4    0.0   
11                               LWSG       0     4        4    4    0.0   
12                               LYDN       0     4        4    4    0.0   
13                               NTQM       0     4        4    4    0.0   
14                               PGLW       0     4        4    4    0.0   
15                               QGRF       0     4        4    4    0.0   
16                               RGMW       0     4        4    4    0.0   
17                               RSFN       0     4        4    4    0.0   
18                               SDKP       0     4        4    4    0.0   
19                               VGSE       0     4        4    4    0.0   
20                               WGHE       0     4        4    4    0.0   
21                               YDFI       0     4        4    4    0.0   
22                               YLRF       0     4        4    4    0.0   
23                               YMRF       0     4        4    4    0.0   
24                               AETF       1     4        4    4    1.0   
25                               AFTS       1     4        4    4    1.0   
26                               AGDV       1     4        4    4    1.0   
27                               AIRS       1     4        4    4    1.0   
28                               AKPF       1     4        4    4    1.0   
29                               ALPF       1     4        4    4    1.0   
...                               ...     ...   ...      ...  ...    ...   
10115  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30    3.0   
10116  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30    2.0   
10117  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
10118  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
10119  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
10120  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
10121  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
10122  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
10123  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
10124  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
10125  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
10126  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
10127  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
10128  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
10129  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
10130  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
10131  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
10132  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
10133  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
10134  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
10135  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
10136  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
10137  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
10138  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
10139  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
10140  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
10141  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
10142  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
10143  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
10144  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

        per_A_x  num_C   per_C_x  num_E    ...      per_N_y   per_Q_y  \
0      0.000000    1.0  0.250000    0.0    ...     0.000000  0.000000   
1      0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
2      0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
3      0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
4      0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
5      0.000000    0.0  0.000000    0.0    ...     0.250000  0.000000   
6      0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
7      0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
8      0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
9      0.000000    0.0  0.000000    1.0    ...     0.000000  0.000000   
10     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
11     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
12     0.000000    0.0  0.000000    0.0    ...     0.250000  0.000000   
13     0.000000    0.0  0.000000    0.0    ...     0.250000  0.250000   
14     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
15     0.000000    0.0  0.000000    0.0    ...     0.000000  0.250000   
16     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
17     0.000000    0.0  0.000000    0.0    ...     0.250000  0.000000   
18     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
19     0.000000    0.0  0.000000    1.0    ...     0.000000  0.000000   
20     0.000000    0.0  0.000000    1.0    ...     0.000000  0.000000   
21     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
22     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
23     0.000000    0.0  0.000000    0.0    ...     0.000000  0.000000   
24     0.250000    0.0  0.000000    1.0    ...     0.000000  0.000000   
25     0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
26     0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
27     0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
28     0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
29     0.250000    0.0  0.000000    0.0    ...     0.000000  0.000000   
...         ...    ...       ...    ...    ...          ...       ...   
10115  0.100000    0.0  0.000000    1.0    ...     0.233333  0.000000   
10116  0.066667    0.0  0.000000    4.0    ...     0.033333  0.133333   
10117  0.300000    0.0  0.000000    0.0    ...     0.033333  0.000000   
10118  0.033333    9.0  0.300000    2.0    ...     0.000000  0.000000   
10119  0.033333    0.0  0.000000    3.0    ...     0.066667  0.066667   
10120  0.066667    2.0  0.066667    3.0    ...     0.100000  0.166667   
10121  0.066667    0.0  0.000000    1.0    ...     0.066667  0.100000   
10122  0.033333    2.0  0.066667    0.0    ...     0.066667  0.000000   
10123  0.033333   11.0  0.366667    0.0    ...     0.000000  0.033333   
10124  0.066667    6.0  0.200000    0.0    ...     0.100000  0.033333   
10125  0.133333    0.0  0.000000    5.0    ...     0.033333  0.000000   
10126  0.166667    0.0  0.000000    3.0    ...     0.100000  0.033333   
10127  0.100000    0.0  0.000000    2.0    ...     0.100000  0.100000   
10128  0.000000    0.0  0.000000    2.0    ...     0.033333  0.000000   
10129  0.100000    0.0  0.000000    8.0    ...     0.033333  0.033333   
10130  0.066667    4.0  0.133333    0.0    ...     0.000000  0.000000   
10131  0.200000    0.0  0.000000    0.0    ...     0.000000  0.000000   
10132  0.066667    0.0  0.000000    3.0    ...     0.100000  0.000000   
10133  0.100000    0.0  0.000000    1.0    ...     0.033333  0.100000   
10134  0.166667    0.0  0.000000    3.0    ...     0.000000  0.033333   
10135  0.100000    0.0  0.000000    0.0    ...     0.000000  0.033333   
10136  0.000000    2.0  0.066667    1.0    ...     0.000000  0.100000   
10137  0.066667    0.0  0.000000    0.0    ...     0.066667  0.166667   
10138  0.066667    0.0  0.000000    0.0    ...     0.066667  0.166667   
10139  0.066667    0.0  0.000000    0.0    ...     0.066667  0.166667   
10140  0.066667    0.0  0.000000    0.0    ...     0.066667  0.166667   
10141  0.000000    1.0  0.033333    0.0    ...     0.000000  0.000000   
10142  0.466667    0.0  0.000000    7.0    ...     0.000000  0.000000   
10143  0.400000    1.0  0.033333    2.0    ...     0.000000  0.000000   
10144  0.400000    1.0  0.033333    2.0    ...     0.000000  0.000000   

        per_P_y   per_S_y   per_R_y   per_T_y   per_W_y   per_V_y   per_Y_y  \
0      0.250000  0.250000  0.000000  0.000000  0.000000  0.000000  0.000000   
1      0.250000  0.250000  0.000000  0.000000  0.000000  0.250000  0.000000   
2      0.250000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000   
3      0.000000  0.000000  0.000000  0.250000  0.000000  0.000000  0.000000   
4      0.250000  0.000000  0.000000  0.000000  0.250000  0.000000  0.000000   
5      0.000000  0.000000  0.250000  0.000000  0.000000  0.000000  0.000000   
6      0.000000  0.000000  0.250000  0.000000  0.000000  0.000000  0.250000   
7      0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000   
8      0.000000  0.250000  0.000000  0.000000  0.250000  0.000000  0.000000   
9      0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000   
10     0.000000  0.000000  0.000000  0.250000  0.250000  0.000000  0.000000   
11     0.000000  0.250000  0.000000  0.000000  0.250000  0.000000  0.000000   
12     0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.250000   
13     0.000000  0.000000  0.000000  0.250000  0.000000  0.000000  0.000000   
14     0.250000  0.000000  0.000000  0.000000  0.250000  0.000000  0.000000   
15     0.000000  0.000000  0.250000  0.000000  0.000000  0.000000  0.000000   
16     0.000000  0.000000  0.250000  0.000000  0.250000  0.000000  0.000000   
17     0.000000  0.250000  0.250000  0.000000  0.000000  0.000000  0.000000   
18     0.250000  0.250000  0.000000  0.000000  0.000000  0.000000  0.000000   
19     0.000000  0.250000  0.000000  0.000000  0.000000  0.250000  0.000000   
20     0.000000  0.000000  0.000000  0.000000  0.250000  0.000000  0.000000   
21     0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.250000   
22     0.000000  0.000000  0.250000  0.000000  0.000000  0.000000  0.250000   
23     0.000000  0.000000  0.250000  0.000000  0.000000  0.000000  0.250000   
24     0.000000  0.000000  0.000000  0.250000  0.000000  0.000000  0.000000   
25     0.000000  0.250000  0.000000  0.250000  0.000000  0.000000  0.000000   
26     0.000000  0.000000  0.000000  0.000000  0.000000  0.250000  0.000000   
27     0.000000  0.250000  0.250000  0.000000  0.000000  0.000000  0.000000   
28     0.250000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000   
29     0.250000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000   
...         ...       ...       ...       ...       ...       ...       ...   
10115  0.000000  0.066667  0.000000  0.100000  0.000000  0.066667  0.033333   
10116  0.000000  0.000000  0.000000  0.066667  0.033333  0.033333  0.033333   
10117  0.066667  0.033333  0.000000  0.066667  0.033333  0.166667  0.066667   
10118  0.133333  0.133333  0.000000  0.100000  0.000000  0.000000  0.000000   
10119  0.200000  0.000000  0.033333  0.033333  0.000000  0.100000  0.066667   
10120  0.000000  0.000000  0.133333  0.000000  0.000000  0.000000  0.033333   
10121  0.000000  0.066667  0.000000  0.066667  0.100000  0.033333  0.033333   
10122  0.033333  0.133333  0.033333  0.066667  0.000000  0.000000  0.066667   
10123  0.066667  0.133333  0.000000  0.100000  0.000000  0.033333  0.000000   
10124  0.033333  0.166667  0.033333  0.033333  0.000000  0.033333  0.000000   
10125  0.000000  0.033333  0.166667  0.100000  0.000000  0.000000  0.033333   
10126  0.033333  0.000000  0.033333  0.033333  0.033333  0.133333  0.000000   
10127  0.166667  0.066667  0.066667  0.066667  0.000000  0.033333  0.033333   
10128  0.033333  0.066667  0.066667  0.066667  0.100000  0.100000  0.033333   
10129  0.000000  0.033333  0.100000  0.033333  0.000000  0.033333  0.033333   
10130  0.100000  0.133333  0.000000  0.100000  0.033333  0.066667  0.066667   
10131  0.100000  0.100000  0.033333  0.033333  0.033333  0.100000  0.000000   
10132  0.033333  0.033333  0.033333  0.033333  0.000000  0.066667  0.000000   
10133  0.000000  0.066667  0.000000  0.033333  0.033333  0.033333  0.066667   
10134  0.000000  0.033333  0.133333  0.066667  0.000000  0.033333  0.133333   
10135  0.100000  0.033333  0.066667  0.100000  0.000000  0.100000  0.066667   
10136  0.033333  0.066667  0.200000  0.033333  0.000000  0.033333  0.000000   
10137  0.100000  0.033333  0.133333  0.000000  0.000000  0.000000  0.000000   
10138  0.100000  0.033333  0.133333  0.000000  0.000000  0.000000  0.000000   
10139  0.100000  0.033333  0.133333  0.000000  0.000000  0.000000  0.000000   
10140  0.100000  0.033333  0.133333  0.000000  0.000000  0.000000  0.000000   
10141  0.366667  0.000000  0.300000  0.000000  0.000000  0.000000  0.033333   
10142  0.000000  0.000000  0.000000  0.000000  0.033333  0.000000  0.000000   
10143  0.000000  0.000000  0.000000  0.000000  0.033333  0.000000  0.000000   
10144  0.000000  0.000000  0.233333  0.000000  0.033333  0.000000  0.000000   

       testPred  
0      0.007797  
1      0.002599  
2      0.031334  
3      0.002652  
4      0.001054  
5      0.020269  
6      0.039302  
7      0.000825  
8      0.003194  
9      0.041583  
10     0.210057  
11     0.005637  
12     0.003981  
13     0.019361  
14     0.005793  
15     0.004671  
16     0.015456  
17     0.012129  
18     0.026276  
19     0.000706  
20     0.007200  
21     0.004430  
22     0.014199  
23     0.008723  
24     0.001721  
25     0.002515  
26     0.001132  
27     0.034738  
28     0.011902  
29     0.011916  
...         ...  
10115  0.008374  
10116  0.004381  
10117  0.012129  
10118  0.002148  
10119  0.001752  
10120  0.015465  
10121  0.005739  
10122  0.044459  
10123  0.003884  
10124  0.001148  
10125  0.006556  
10126  0.009148  
10127  0.009225  
10128  0.002634  
10129  0.005612  
10130  0.003837  
10131  0.020465  
10132  0.001434  
10133  0.001645  
10134  0.022678  
10135  0.007942  
10136  0.016428  
10137  0.640396  
10138  0.654751  
10139  0.640396  
10140  0.654751  
10141  0.411098  
10142  0.015545  
10143  0.932408  
10144  0.980807  

[10145 rows x 68 columns]
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 96, in <module>
    data2 = data.merge(data_xgb[['testPred',]+features],on='seq')
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 'seq'
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 97, in <module>
    die
NameError: name 'die' is not defined
>>> data2
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9843  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9844  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9845  NNDITAENNNIYKAAKDVTTSLSKVLKNIN       1    30       11   30    3.0   
9846  NTLDTLYKEQIAEDIVWDIIDELEQIALQQ       1    30       12   30    2.0   
9847  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9848  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9849  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9850  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9851  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9852  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9853  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9854  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9855  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9856  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9857  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9858  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9859  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9860  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9861  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9862  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9863  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9864  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9865  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9866  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9867  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9868  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9869  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9870  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9871  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9872  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E    ...        per_R  num_T     per_T  \
0     0.000000    1.0  0.250000    0.0    ...     0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
2     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
4     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
7     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
8     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
9     0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
10    0.000000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
11    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
13    0.000000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
14    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
16    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
20    0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
22    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
23    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
24    0.250000    0.0  0.000000    1.0    ...     0.000000    1.0  0.250000   
25    0.250000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
26    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
27    0.250000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
28    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
29    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
...        ...    ...       ...    ...    ...          ...    ...       ...   
9843  0.033333    1.0  0.033333    0.0    ...     0.033333    0.0  0.000000   
9844  0.033333    1.0  0.033333    0.0    ...     0.033333    0.0  0.000000   
9845  0.100000    0.0  0.000000    1.0    ...     0.000000    3.0  0.100000   
9846  0.066667    0.0  0.000000    4.0    ...     0.000000    2.0  0.066667   
9847  0.300000    0.0  0.000000    0.0    ...     0.000000    2.0  0.066667   
9848  0.033333    9.0  0.300000    2.0    ...     0.000000    3.0  0.100000   
9849  0.033333    0.0  0.000000    3.0    ...     0.033333    1.0  0.033333   
9850  0.066667    2.0  0.066667    3.0    ...     0.133333    0.0  0.000000   
9851  0.066667    0.0  0.000000    1.0    ...     0.000000    2.0  0.066667   
9852  0.033333    2.0  0.066667    0.0    ...     0.033333    2.0  0.066667   
9853  0.033333   11.0  0.366667    0.0    ...     0.000000    3.0  0.100000   
9854  0.066667    6.0  0.200000    0.0    ...     0.033333    1.0  0.033333   
9855  0.133333    0.0  0.000000    5.0    ...     0.166667    3.0  0.100000   
9856  0.166667    0.0  0.000000    3.0    ...     0.033333    1.0  0.033333   
9857  0.100000    0.0  0.000000    2.0    ...     0.066667    2.0  0.066667   
9858  0.000000    0.0  0.000000    2.0    ...     0.066667    2.0  0.066667   
9859  0.100000    0.0  0.000000    8.0    ...     0.100000    1.0  0.033333   
9860  0.066667    4.0  0.133333    0.0    ...     0.000000    3.0  0.100000   
9861  0.200000    0.0  0.000000    0.0    ...     0.033333    1.0  0.033333   
9862  0.066667    0.0  0.000000    3.0    ...     0.033333    1.0  0.033333   
9863  0.100000    0.0  0.000000    1.0    ...     0.000000    1.0  0.033333   
9864  0.166667    0.0  0.000000    3.0    ...     0.133333    2.0  0.066667   
9865  0.100000    0.0  0.000000    0.0    ...     0.066667    3.0  0.100000   
9866  0.000000    2.0  0.066667    1.0    ...     0.200000    1.0  0.033333   
9867  0.066667    0.0  0.000000    0.0    ...     0.133333    0.0  0.000000   
9868  0.000000    1.0  0.033333    0.0    ...     0.300000    0.0  0.000000   
9869  0.066667    0.0  0.000000    0.0    ...     0.133333    0.0  0.000000   
9870  0.466667    0.0  0.000000    7.0    ...     0.000000    0.0  0.000000   
9871  0.400000    1.0  0.033333    2.0    ...     0.000000    0.0  0.000000   
9872  0.400000    1.0  0.033333    2.0    ...     0.233333    0.0  0.000000   

      num_W     per_W  num_V     per_V  num_Y     per_Y  testPred  
0       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.007797  
1       0.0  0.000000    1.0  0.250000    0.0  0.000000  0.002599  
2       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.031334  
3       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.002652  
4       1.0  0.250000    0.0  0.000000    0.0  0.000000  0.001054  
5       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.020269  
6       0.0  0.000000    0.0  0.000000    1.0  0.250000  0.039302  
7       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.000825  
8       1.0  0.250000    0.0  0.000000    0.0  0.000000  0.003194  
9       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.041583  
10      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.210057  
11      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.005637  
12      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.003981  
13      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.019361  
14      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.005793  
15      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.004671  
16      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.015456  
17      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.012129  
18      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.026276  
19      0.0  0.000000    1.0  0.250000    0.0  0.000000  0.000706  
20      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.007200  
21      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.004430  
22      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.014199  
23      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.008723  
24      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.001721  
25      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.002515  
26      0.0  0.000000    1.0  0.250000    0.0  0.000000  0.001132  
27      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.034738  
28      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.011902  
29      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.011916  
...     ...       ...    ...       ...    ...       ...       ...  
9843    2.0  0.066667    4.0  0.133333    0.0  0.000000  0.643268  
9844    2.0  0.066667    4.0  0.133333    0.0  0.000000  0.072823  
9845    0.0  0.000000    2.0  0.066667    1.0  0.033333  0.008374  
9846    1.0  0.033333    1.0  0.033333    1.0  0.033333  0.004381  
9847    1.0  0.033333    5.0  0.166667    2.0  0.066667  0.012129  
9848    0.0  0.000000    0.0  0.000000    0.0  0.000000  0.002148  
9849    0.0  0.000000    3.0  0.100000    2.0  0.066667  0.001752  
9850    0.0  0.000000    0.0  0.000000    1.0  0.033333  0.015465  
9851    3.0  0.100000    1.0  0.033333    1.0  0.033333  0.005739  
9852    0.0  0.000000    0.0  0.000000    2.0  0.066667  0.044459  
9853    0.0  0.000000    1.0  0.033333    0.0  0.000000  0.003884  
9854    0.0  0.000000    1.0  0.033333    0.0  0.000000  0.001148  
9855    0.0  0.000000    0.0  0.000000    1.0  0.033333  0.006556  
9856    1.0  0.033333    4.0  0.133333    0.0  0.000000  0.009148  
9857    0.0  0.000000    1.0  0.033333    1.0  0.033333  0.009225  
9858    3.0  0.100000    3.0  0.100000    1.0  0.033333  0.002634  
9859    0.0  0.000000    1.0  0.033333    1.0  0.033333  0.005612  
9860    1.0  0.033333    2.0  0.066667    2.0  0.066667  0.003837  
9861    1.0  0.033333    3.0  0.100000    0.0  0.000000  0.020465  
9862    0.0  0.000000    2.0  0.066667    0.0  0.000000  0.001434  
9863    1.0  0.033333    1.0  0.033333    2.0  0.066667  0.001645  
9864    0.0  0.000000    1.0  0.033333    4.0  0.133333  0.022678  
9865    0.0  0.000000    3.0  0.100000    2.0  0.066667  0.007942  
9866    0.0  0.000000    1.0  0.033333    0.0  0.000000  0.016428  
9867    0.0  0.000000    0.0  0.000000    0.0  0.000000  0.640396  
9868    0.0  0.000000    0.0  0.000000    1.0  0.033333  0.411098  
9869    0.0  0.000000    0.0  0.000000    0.0  0.000000  0.654751  
9870    1.0  0.033333    0.0  0.000000    0.0  0.000000  0.015545  
9871    1.0  0.033333    0.0  0.000000    0.0  0.000000  0.932408  
9872    1.0  0.033333    0.0  0.000000    0.0  0.000000  0.980807  

[9873 rows x 46 columns]
>>> 
KeyboardInterrupt
>>> pd.unique(data.seq)
array(['SPCG', 'VPSG', 'ADKP', ..., 'WEAALAEALAEALAEHLAEALAEALEALAA',
       'WEAKLAKALAKALAKHLAKALAKALKACEA', 'WEARLARALARALARHLARALARALRACEA'], dtype=object)
>>> len(pd.unique(data.seq))
9654
>>> 
KeyboardInterrupt
>>> data
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E    ...     num_R     per_R  num_T  \
0     0.000000    1.0  0.250000    0.0    ...       0.0  0.000000    0.0   
1     0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
2     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
3     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
4     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
5     0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
6     0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
7     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
8     0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
9     0.000000    0.0  0.000000    1.0    ...       0.0  0.000000    0.0   
10    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
11    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
12    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
13    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
14    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
15    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
16    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
17    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
18    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
19    0.000000    0.0  0.000000    1.0    ...       0.0  0.000000    0.0   
20    0.000000    0.0  0.000000    1.0    ...       0.0  0.000000    0.0   
21    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
22    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
23    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
24    0.250000    0.0  0.000000    1.0    ...       0.0  0.000000    1.0   
25    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
26    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
27    0.250000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
28    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
29    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
...        ...    ...       ...    ...    ...       ...       ...    ...   
9697  0.300000    0.0  0.000000    0.0    ...       0.0  0.000000    2.0   
9698  0.033333    9.0  0.300000    2.0    ...       0.0  0.000000    3.0   
9699  0.033333    0.0  0.000000    3.0    ...       1.0  0.033333    1.0   
9700  0.066667    2.0  0.066667    3.0    ...       4.0  0.133333    0.0   
9701  0.066667    0.0  0.000000    1.0    ...       0.0  0.000000    2.0   
9702  0.033333    2.0  0.066667    0.0    ...       1.0  0.033333    2.0   
9703  0.033333   11.0  0.366667    0.0    ...       0.0  0.000000    3.0   
9704  0.066667    6.0  0.200000    0.0    ...       1.0  0.033333    1.0   
9705  0.000000    6.0  0.200000    1.0    ...       3.0  0.100000    0.0   
9706  0.133333    0.0  0.000000    5.0    ...       5.0  0.166667    3.0   
9707  0.166667    0.0  0.000000    3.0    ...       1.0  0.033333    1.0   
9708  0.100000    0.0  0.000000    2.0    ...       2.0  0.066667    2.0   
9709  0.000000    0.0  0.000000    2.0    ...       2.0  0.066667    2.0   
9710  0.100000    0.0  0.000000    8.0    ...       3.0  0.100000    1.0   
9711  0.066667    4.0  0.133333    0.0    ...       0.0  0.000000    3.0   
9712  0.200000    0.0  0.000000    0.0    ...       1.0  0.033333    1.0   
9713  0.066667    0.0  0.000000    3.0    ...       1.0  0.033333    1.0   
9714  0.100000    0.0  0.000000    1.0    ...       0.0  0.000000    1.0   
9715  0.166667    0.0  0.000000    3.0    ...       4.0  0.133333    2.0   
9716  0.000000    6.0  0.200000    2.0    ...       2.0  0.066667    1.0   
9717  0.100000    0.0  0.000000    0.0    ...       2.0  0.066667    3.0   
9718  0.000000    2.0  0.066667    1.0    ...       6.0  0.200000    1.0   
9719  0.066667    0.0  0.000000    0.0    ...       1.0  0.033333    2.0   
9720  0.066667    0.0  0.000000    0.0    ...       4.0  0.133333    0.0   
9721  0.000000    1.0  0.033333    0.0    ...       9.0  0.300000    0.0   
9722  0.033333    1.0  0.033333    0.0    ...       1.0  0.033333    0.0   
9723  0.066667    0.0  0.000000    0.0    ...       4.0  0.133333    0.0   
9724  0.466667    0.0  0.000000    7.0    ...       0.0  0.000000    0.0   
9725  0.400000    1.0  0.033333    2.0    ...       0.0  0.000000    0.0   
9726  0.400000    1.0  0.033333    2.0    ...       7.0  0.233333    0.0   

         per_T  num_W     per_W  num_V     per_V  num_Y     per_Y  
0     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
1     0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000  
2     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
3     0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
4     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
5     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
6     0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
7     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
8     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
9     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
10    0.250000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
11    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
12    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
13    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
14    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
15    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
16    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
17    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
18    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
19    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000  
20    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
21    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
22    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
23    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
24    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
25    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
26    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000  
27    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
28    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
29    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
...        ...    ...       ...    ...       ...    ...       ...  
9697  0.066667    1.0  0.033333    5.0  0.166667    2.0  0.066667  
9698  0.100000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
9699  0.033333    0.0  0.000000    3.0  0.100000    2.0  0.066667  
9700  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333  
9701  0.066667    3.0  0.100000    1.0  0.033333    1.0  0.033333  
9702  0.066667    0.0  0.000000    0.0  0.000000    2.0  0.066667  
9703  0.100000    0.0  0.000000    1.0  0.033333    0.0  0.000000  
9704  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000  
9705  0.000000    0.0  0.000000    2.0  0.066667    1.0  0.033333  
9706  0.100000    0.0  0.000000    0.0  0.000000    1.0  0.033333  
9707  0.033333    1.0  0.033333    4.0  0.133333    0.0  0.000000  
9708  0.066667    0.0  0.000000    1.0  0.033333    1.0  0.033333  
9709  0.066667    3.0  0.100000    3.0  0.100000    1.0  0.033333  
9710  0.033333    0.0  0.000000    1.0  0.033333    1.0  0.033333  
9711  0.100000    1.0  0.033333    2.0  0.066667    2.0  0.066667  
9712  0.033333    1.0  0.033333    3.0  0.100000    0.0  0.000000  
9713  0.033333    0.0  0.000000    2.0  0.066667    0.0  0.000000  
9714  0.033333    1.0  0.033333    1.0  0.033333    2.0  0.066667  
9715  0.066667    0.0  0.000000    1.0  0.033333    4.0  0.133333  
9716  0.033333    4.0  0.133333    1.0  0.033333    1.0  0.033333  
9717  0.100000    0.0  0.000000    3.0  0.100000    2.0  0.066667  
9718  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000  
9719  0.066667    1.0  0.033333    1.0  0.033333    1.0  0.033333  
9720  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
9721  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333  
9722  0.000000    2.0  0.066667    4.0  0.133333    0.0  0.000000  
9723  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
9724  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000  
9725  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000  
9726  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000  

[9727 rows x 45 columns]
>>> 
KeyboardInterrupt
>>> data.merge(data_xgb[['testPred','seq',]],on=index)

Traceback (most recent call last):
  File "<pyshell#77>", line 1, in <module>
    data.merge(data_xgb[['testPred','seq',]],on=index)
NameError: name 'index' is not defined
>>> data.merge(data_xgb[['testPred','seq',]],on=ind)

Traceback (most recent call last):
  File "<pyshell#78>", line 1, in <module>
    data.merge(data_xgb[['testPred','seq',]],on=ind)
NameError: name 'ind' is not defined
>>> data.merge(data_xgb[['testPred','seq',]],on=id)

Traceback (most recent call last):
  File "<pyshell#79>", line 1, in <module>
    data.merge(data_xgb[['testPred','seq',]],on=id)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 4722, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 53, in merge
    copy=copy, indicator=indicator)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 558, in __init__
    self.join_names) = self._get_merge_keys()
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/reshape/merge.py", line 810, in _get_merge_keys
    right_keys.append(right[rk]._values)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1964, in __getitem__
    return self._getitem_column(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/frame.py", line 1971, in _getitem_column
    return self._get_item_cache(key)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1645, in _get_item_cache
    values = self._data.get(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/internals.py", line 3590, in get
    loc = self.items.get_loc(item)
  File "/usr/local/lib/python2.7/dist-packages/pandas/core/indexes/base.py", line 2444, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas/_libs/index.pyx", line 132, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)
  File "pandas/_libs/index.pyx", line 154, in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1210, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)
  File "pandas/_libs/hashtable_class_helper.pxi", line 1218, in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)
KeyError: 140111079904144
>>> data
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E    ...     num_R     per_R  num_T  \
0     0.000000    1.0  0.250000    0.0    ...       0.0  0.000000    0.0   
1     0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
2     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
3     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
4     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
5     0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
6     0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
7     0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
8     0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
9     0.000000    0.0  0.000000    1.0    ...       0.0  0.000000    0.0   
10    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
11    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
12    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
13    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
14    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
15    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
16    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
17    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
18    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
19    0.000000    0.0  0.000000    1.0    ...       0.0  0.000000    0.0   
20    0.000000    0.0  0.000000    1.0    ...       0.0  0.000000    0.0   
21    0.000000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
22    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
23    0.000000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
24    0.250000    0.0  0.000000    1.0    ...       0.0  0.000000    1.0   
25    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    1.0   
26    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
27    0.250000    0.0  0.000000    0.0    ...       1.0  0.250000    0.0   
28    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
29    0.250000    0.0  0.000000    0.0    ...       0.0  0.000000    0.0   
...        ...    ...       ...    ...    ...       ...       ...    ...   
9697  0.300000    0.0  0.000000    0.0    ...       0.0  0.000000    2.0   
9698  0.033333    9.0  0.300000    2.0    ...       0.0  0.000000    3.0   
9699  0.033333    0.0  0.000000    3.0    ...       1.0  0.033333    1.0   
9700  0.066667    2.0  0.066667    3.0    ...       4.0  0.133333    0.0   
9701  0.066667    0.0  0.000000    1.0    ...       0.0  0.000000    2.0   
9702  0.033333    2.0  0.066667    0.0    ...       1.0  0.033333    2.0   
9703  0.033333   11.0  0.366667    0.0    ...       0.0  0.000000    3.0   
9704  0.066667    6.0  0.200000    0.0    ...       1.0  0.033333    1.0   
9705  0.000000    6.0  0.200000    1.0    ...       3.0  0.100000    0.0   
9706  0.133333    0.0  0.000000    5.0    ...       5.0  0.166667    3.0   
9707  0.166667    0.0  0.000000    3.0    ...       1.0  0.033333    1.0   
9708  0.100000    0.0  0.000000    2.0    ...       2.0  0.066667    2.0   
9709  0.000000    0.0  0.000000    2.0    ...       2.0  0.066667    2.0   
9710  0.100000    0.0  0.000000    8.0    ...       3.0  0.100000    1.0   
9711  0.066667    4.0  0.133333    0.0    ...       0.0  0.000000    3.0   
9712  0.200000    0.0  0.000000    0.0    ...       1.0  0.033333    1.0   
9713  0.066667    0.0  0.000000    3.0    ...       1.0  0.033333    1.0   
9714  0.100000    0.0  0.000000    1.0    ...       0.0  0.000000    1.0   
9715  0.166667    0.0  0.000000    3.0    ...       4.0  0.133333    2.0   
9716  0.000000    6.0  0.200000    2.0    ...       2.0  0.066667    1.0   
9717  0.100000    0.0  0.000000    0.0    ...       2.0  0.066667    3.0   
9718  0.000000    2.0  0.066667    1.0    ...       6.0  0.200000    1.0   
9719  0.066667    0.0  0.000000    0.0    ...       1.0  0.033333    2.0   
9720  0.066667    0.0  0.000000    0.0    ...       4.0  0.133333    0.0   
9721  0.000000    1.0  0.033333    0.0    ...       9.0  0.300000    0.0   
9722  0.033333    1.0  0.033333    0.0    ...       1.0  0.033333    0.0   
9723  0.066667    0.0  0.000000    0.0    ...       4.0  0.133333    0.0   
9724  0.466667    0.0  0.000000    7.0    ...       0.0  0.000000    0.0   
9725  0.400000    1.0  0.033333    2.0    ...       0.0  0.000000    0.0   
9726  0.400000    1.0  0.033333    2.0    ...       7.0  0.233333    0.0   

         per_T  num_W     per_W  num_V     per_V  num_Y     per_Y  
0     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
1     0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000  
2     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
3     0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
4     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
5     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
6     0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
7     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
8     0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
9     0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
10    0.250000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
11    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
12    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
13    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
14    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
15    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
16    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
17    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
18    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
19    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000  
20    0.000000    1.0  0.250000    0.0  0.000000    0.0  0.000000  
21    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
22    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
23    0.000000    0.0  0.000000    0.0  0.000000    1.0  0.250000  
24    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
25    0.250000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
26    0.000000    0.0  0.000000    1.0  0.250000    0.0  0.000000  
27    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
28    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
29    0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
...        ...    ...       ...    ...       ...    ...       ...  
9697  0.066667    1.0  0.033333    5.0  0.166667    2.0  0.066667  
9698  0.100000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
9699  0.033333    0.0  0.000000    3.0  0.100000    2.0  0.066667  
9700  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333  
9701  0.066667    3.0  0.100000    1.0  0.033333    1.0  0.033333  
9702  0.066667    0.0  0.000000    0.0  0.000000    2.0  0.066667  
9703  0.100000    0.0  0.000000    1.0  0.033333    0.0  0.000000  
9704  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000  
9705  0.000000    0.0  0.000000    2.0  0.066667    1.0  0.033333  
9706  0.100000    0.0  0.000000    0.0  0.000000    1.0  0.033333  
9707  0.033333    1.0  0.033333    4.0  0.133333    0.0  0.000000  
9708  0.066667    0.0  0.000000    1.0  0.033333    1.0  0.033333  
9709  0.066667    3.0  0.100000    3.0  0.100000    1.0  0.033333  
9710  0.033333    0.0  0.000000    1.0  0.033333    1.0  0.033333  
9711  0.100000    1.0  0.033333    2.0  0.066667    2.0  0.066667  
9712  0.033333    1.0  0.033333    3.0  0.100000    0.0  0.000000  
9713  0.033333    0.0  0.000000    2.0  0.066667    0.0  0.000000  
9714  0.033333    1.0  0.033333    1.0  0.033333    2.0  0.066667  
9715  0.066667    0.0  0.000000    1.0  0.033333    4.0  0.133333  
9716  0.033333    4.0  0.133333    1.0  0.033333    1.0  0.033333  
9717  0.100000    0.0  0.000000    3.0  0.100000    2.0  0.066667  
9718  0.033333    0.0  0.000000    1.0  0.033333    0.0  0.000000  
9719  0.066667    1.0  0.033333    1.0  0.033333    1.0  0.033333  
9720  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
9721  0.000000    0.0  0.000000    0.0  0.000000    1.0  0.033333  
9722  0.000000    2.0  0.066667    4.0  0.133333    0.0  0.000000  
9723  0.000000    0.0  0.000000    0.0  0.000000    0.0  0.000000  
9724  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000  
9725  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000  
9726  0.000000    1.0  0.033333    0.0  0.000000    0.0  0.000000  

[9727 rows x 45 columns]
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}

Traceback (most recent call last):
  File "/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py", line 97, in <module>
    die
NameError: name 'die' is not defined
>>> data2
                                 seq  source  size  type_aa  len  num_A  \
0                               SPCG      -1     4        4    4    0.0   
1                               VPSG      -1     4        4    4    0.0   
2                               ADKP       0     4        4    4    1.0   
3                               AGLT       0     4        4    4    1.0   
4                               APGW       0     4        4    4    1.0   
5                               FLRN       0     4        4    4    0.0   
6                               FYRI       0     4        4    4    0.0   
7                               GFAD       0     4        4    4    1.0   
8                               GSWD       0     4        4    4    0.0   
9                               ILME       0     4        4    4    0.0   
10                              LWKT       0     4        4    4    0.0   
11                              LWSG       0     4        4    4    0.0   
12                              LYDN       0     4        4    4    0.0   
13                              NTQM       0     4        4    4    0.0   
14                              PGLW       0     4        4    4    0.0   
15                              QGRF       0     4        4    4    0.0   
16                              RGMW       0     4        4    4    0.0   
17                              RSFN       0     4        4    4    0.0   
18                              SDKP       0     4        4    4    0.0   
19                              VGSE       0     4        4    4    0.0   
20                              WGHE       0     4        4    4    0.0   
21                              YDFI       0     4        4    4    0.0   
22                              YLRF       0     4        4    4    0.0   
23                              YMRF       0     4        4    4    0.0   
24                              AETF       1     4        4    4    1.0   
25                              AFTS       1     4        4    4    1.0   
26                              AGDV       1     4        4    4    1.0   
27                              AIRS       1     4        4    4    1.0   
28                              AKPF       1     4        4    4    1.0   
29                              ALPF       1     4        4    4    1.0   
...                              ...     ...   ...      ...  ...    ...   
9697  PAIYIGATVGPSVWAYLVALVGAAAVTAAN       1    30       11   30    9.0   
9698  PCEKCTSGCKCPSKDECAKTCSKPCSCCPT       1    30        9   30    1.0   
9699  PFPPTPPGEEAPVEDLIRFYNDLQQYLNVV       1    30       14   30    1.0   
9700  QQARQNLQNLYINRCLREICQELKEIRAML       1    30       11   30    2.0   
9701  QTNWQKLEVFWAKHMWNFISGIQYLAGLST       1    30       16   30    2.0   
9702  RPYHCSYCNFSFKTKGNLTKHMKSKAHSKK       1    30       14   30    1.0   
9703  SCCPCCPSGCTKCASGCVCKGKTCDTSCCQ       1    30       10   30    1.0   
9704  SCNNSCQSHSDCASHCICTFRGCGAVNGLP       1    30       15   30    2.0   
9705  SCSGRDSRCPPVCCMGLMCSRGKCVSIYGE       1    30       13   30    0.0   
9706  SEKAAEEAYTRTTRALHERFDRLERMLDDN       1    30       13   30    4.0   
9707  TAFVEPFVILLILIANAIVGVWQERNAENA       1    30       13   30    5.0   
9708  TPVERQTIYSQAPSLNPNLILAAPPKERNQ       1    30       13   30    3.0   
9709  TVFTSWEEYLDWVGSGDLMPWNLVRIGLLR       1    30       15   30    0.0   
9710  VEKLTADAELQRLKNERHEEAELERLLSEY       1    30       13   30    3.0   
9711  VIHCDAATICPDGTTCSLSPYGVWYCSPFS       1    30       14   30    2.0   
9712  VKGRIDAPDFPSSPAILGKAATDVVAAWKS       1    30       13   30    6.0   
9713  VTVDDDDDDNDPENRIAKKMLLEEIKANLS       1    30       13   30    2.0   
9714  YAEGTFISDYSIAMDKIHQQDFVNWLLAQK       1    30       17   30    3.0   
9715  YAEKVAQEKGFLYRLTSRYRHYAAFERATF       1    30       13   30    5.0   
9716  YCQKWMWTCDSERKCCEGMVCRLWCKKKLW       1    30       14   30    0.0   
9717  ADVFDRGGPYLQRGVADLVPTATLLDTYSP       2    30       12   30    3.0   
9718  CGGKDCERRFSRSDQLKRHQRRHTGVKPFQ       2    30       14   30    0.0   
9719  GWTLNSAGYLLGPHAVGNHRSFSDKNGLTS       2    30       15   30    2.0   
9720  LLIILRRRIRKQAHAHSKNHQQQNPHQPPM       2    30       11   30    2.0   
9721  MRRIRPRPPRLPRPRPRPLPFPRPGGCYPG       2    30        9   30    0.0   
9722  MVKSKIGSWILVLFVAMWSDVGLCKKRPKP       2    30       14   30    1.0   
9723  NHQQQNPHQPPMLLIILRRRIRKQAHAHSK       2    30       11   30    2.0   
9724  WEAALAEALAEALAEHLAEALAEALEALAA       2    30        5   30   14.0   
9725  WEAKLAKALAKALAKHLAKALAKALKACEA       2    30        7   30   12.0   
9726  WEARLARALARALARHLARALARALRACEA       2    30        7   30   12.0   

         per_A  num_C     per_C  num_E    ...        per_R  num_T     per_T  \
0     0.000000    1.0  0.250000    0.0    ...     0.000000    0.0  0.000000   
1     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
2     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
3     0.250000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
4     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
5     0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
6     0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
7     0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
8     0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
9     0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
10    0.000000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
11    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
12    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
13    0.000000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
14    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
15    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
16    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
17    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
18    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
19    0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
20    0.000000    0.0  0.000000    1.0    ...     0.000000    0.0  0.000000   
21    0.000000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
22    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
23    0.000000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
24    0.250000    0.0  0.000000    1.0    ...     0.000000    1.0  0.250000   
25    0.250000    0.0  0.000000    0.0    ...     0.000000    1.0  0.250000   
26    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
27    0.250000    0.0  0.000000    0.0    ...     0.250000    0.0  0.000000   
28    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
29    0.250000    0.0  0.000000    0.0    ...     0.000000    0.0  0.000000   
...        ...    ...       ...    ...    ...          ...    ...       ...   
9697  0.300000    0.0  0.000000    0.0    ...     0.000000    2.0  0.066667   
9698  0.033333    9.0  0.300000    2.0    ...     0.000000    3.0  0.100000   
9699  0.033333    0.0  0.000000    3.0    ...     0.033333    1.0  0.033333   
9700  0.066667    2.0  0.066667    3.0    ...     0.133333    0.0  0.000000   
9701  0.066667    0.0  0.000000    1.0    ...     0.000000    2.0  0.066667   
9702  0.033333    2.0  0.066667    0.0    ...     0.033333    2.0  0.066667   
9703  0.033333   11.0  0.366667    0.0    ...     0.000000    3.0  0.100000   
9704  0.066667    6.0  0.200000    0.0    ...     0.033333    1.0  0.033333   
9705  0.000000    6.0  0.200000    1.0    ...     0.100000    0.0  0.000000   
9706  0.133333    0.0  0.000000    5.0    ...     0.166667    3.0  0.100000   
9707  0.166667    0.0  0.000000    3.0    ...     0.033333    1.0  0.033333   
9708  0.100000    0.0  0.000000    2.0    ...     0.066667    2.0  0.066667   
9709  0.000000    0.0  0.000000    2.0    ...     0.066667    2.0  0.066667   
9710  0.100000    0.0  0.000000    8.0    ...     0.100000    1.0  0.033333   
9711  0.066667    4.0  0.133333    0.0    ...     0.000000    3.0  0.100000   
9712  0.200000    0.0  0.000000    0.0    ...     0.033333    1.0  0.033333   
9713  0.066667    0.0  0.000000    3.0    ...     0.033333    1.0  0.033333   
9714  0.100000    0.0  0.000000    1.0    ...     0.000000    1.0  0.033333   
9715  0.166667    0.0  0.000000    3.0    ...     0.133333    2.0  0.066667   
9716  0.000000    6.0  0.200000    2.0    ...     0.066667    1.0  0.033333   
9717  0.100000    0.0  0.000000    0.0    ...     0.066667    3.0  0.100000   
9718  0.000000    2.0  0.066667    1.0    ...     0.200000    1.0  0.033333   
9719  0.066667    0.0  0.000000    0.0    ...     0.033333    2.0  0.066667   
9720  0.066667    0.0  0.000000    0.0    ...     0.133333    0.0  0.000000   
9721  0.000000    1.0  0.033333    0.0    ...     0.300000    0.0  0.000000   
9722  0.033333    1.0  0.033333    0.0    ...     0.033333    0.0  0.000000   
9723  0.066667    0.0  0.000000    0.0    ...     0.133333    0.0  0.000000   
9724  0.466667    0.0  0.000000    7.0    ...     0.000000    0.0  0.000000   
9725  0.400000    1.0  0.033333    2.0    ...     0.000000    0.0  0.000000   
9726  0.400000    1.0  0.033333    2.0    ...     0.233333    0.0  0.000000   

      num_W     per_W  num_V     per_V  num_Y     per_Y  testPred  
0       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.007797  
1       0.0  0.000000    1.0  0.250000    0.0  0.000000  0.002599  
2       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.031334  
3       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.002652  
4       1.0  0.250000    0.0  0.000000    0.0  0.000000  0.001054  
5       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.020269  
6       0.0  0.000000    0.0  0.000000    1.0  0.250000  0.039302  
7       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.000825  
8       1.0  0.250000    0.0  0.000000    0.0  0.000000  0.003194  
9       0.0  0.000000    0.0  0.000000    0.0  0.000000  0.041583  
10      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.210057  
11      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.005637  
12      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.003981  
13      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.019361  
14      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.005793  
15      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.004671  
16      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.015456  
17      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.012129  
18      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.026276  
19      0.0  0.000000    1.0  0.250000    0.0  0.000000  0.000706  
20      1.0  0.250000    0.0  0.000000    0.0  0.000000  0.007200  
21      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.004430  
22      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.014199  
23      0.0  0.000000    0.0  0.000000    1.0  0.250000  0.008723  
24      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.001721  
25      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.002515  
26      0.0  0.000000    1.0  0.250000    0.0  0.000000  0.001132  
27      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.034738  
28      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.011902  
29      0.0  0.000000    0.0  0.000000    0.0  0.000000  0.011916  
...     ...       ...    ...       ...    ...       ...       ...  
9697    1.0  0.033333    5.0  0.166667    2.0  0.066667  0.012129  
9698    0.0  0.000000    0.0  0.000000    0.0  0.000000  0.002148  
9699    0.0  0.000000    3.0  0.100000    2.0  0.066667  0.001752  
9700    0.0  0.000000    0.0  0.000000    1.0  0.033333  0.015465  
9701    3.0  0.100000    1.0  0.033333    1.0  0.033333  0.005739  
9702    0.0  0.000000    0.0  0.000000    2.0  0.066667  0.044459  
9703    0.0  0.000000    1.0  0.033333    0.0  0.000000  0.003884  
9704    0.0  0.000000    1.0  0.033333    0.0  0.000000  0.001148  
9705    0.0  0.000000    2.0  0.066667    1.0  0.033333  0.000774  
9706    0.0  0.000000    0.0  0.000000    1.0  0.033333  0.006556  
9707    1.0  0.033333    4.0  0.133333    0.0  0.000000  0.009148  
9708    0.0  0.000000    1.0  0.033333    1.0  0.033333  0.009225  
9709    3.0  0.100000    3.0  0.100000    1.0  0.033333  0.002634  
9710    0.0  0.000000    1.0  0.033333    1.0  0.033333  0.005612  
9711    1.0  0.033333    2.0  0.066667    2.0  0.066667  0.003837  
9712    1.0  0.033333    3.0  0.100000    0.0  0.000000  0.020465  
9713    0.0  0.000000    2.0  0.066667    0.0  0.000000  0.001434  
9714    1.0  0.033333    1.0  0.033333    2.0  0.066667  0.001645  
9715    0.0  0.000000    1.0  0.033333    4.0  0.133333  0.022678  
9716    4.0  0.133333    1.0  0.033333    1.0  0.033333  0.013341  
9717    0.0  0.000000    3.0  0.100000    2.0  0.066667  0.007942  
9718    0.0  0.000000    1.0  0.033333    0.0  0.000000  0.016428  
9719    1.0  0.033333    1.0  0.033333    1.0  0.033333  0.003055  
9720    0.0  0.000000    0.0  0.000000    0.0  0.000000  0.640396  
9721    0.0  0.000000    0.0  0.000000    1.0  0.033333  0.411098  
9722    2.0  0.066667    4.0  0.133333    0.0  0.000000  0.072823  
9723    0.0  0.000000    0.0  0.000000    0.0  0.000000  0.654751  
9724    1.0  0.033333    0.0  0.000000    0.0  0.000000  0.015545  
9725    1.0  0.033333    0.0  0.000000    0.0  0.000000  0.932408  
9726    1.0  0.033333    0.0  0.000000    0.0  0.000000  0.980807  

[9727 rows x 46 columns]
>>> 
 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.624229 0.608428
0.848161353153 0.837422100943 0.814910569106
0.804147 0.795478
0.867833809014 0.840329981194 0.830352303523
0.824709 0.816033
0.883466133665 0.856918595157 0.852466124661
SAVED


 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.785298 0.778006
0.914424066721 0.873206938877 0.826184281843
0.905586 0.904933
0.952248001993 0.892450573938 0.857425474255
0.918609 0.911614
0.975251120175 0.915450220988 0.872531165312
SAVED

0.943797 0.929085
0.988103191109 0.914417712784 0.876195121951
SAVED

0.947053 0.926002
0.991502279364 0.917193895559 0.878113821138
SAVED


 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945

 RESTART: /media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/CNN_layer6_avg_pooling.py 
removing these number of non-cannonical peptides 1129
{'A': 13809, 'C': 6853, 'E': 6730, 'D': 6110, 'G': 13341, 'F': 7237, 'I': 6698, 'H': 3514, 'K': 8574, 'M': 5073, 'L': 21706, 'N': 4807, 'Q': 5262, 'P': 9896, 'S': 11657, 'R': 9044, 'T': 7066, 'W': 3261, 'V': 9026, 'Y': 4492}
{'A': 0.084000000000000005, 'C': 0.042000000000000003, 'E': 0.041000000000000002, 'D': 0.036999999999999998, 'G': 0.081000000000000003, 'F': 0.043999999999999997, 'I': 0.041000000000000002, 'H': 0.021000000000000001, 'K': 0.051999999999999998, 'M': 0.031, 'L': 0.13200000000000001, 'N': 0.029000000000000001, 'Q': 0.032000000000000001, 'P': 0.059999999999999998, 'S': 0.070999999999999994, 'R': 0.055, 'T': 0.042999999999999997, 'W': 0.02, 'V': 0.055, 'Y': 0.027}
Couldn't import dot_parser, loading of dot files will not be possible.
869318
20
TEST IS 2
train_val_test size: 9727
train+val  size : 7782
size of different sets: 10953 1946 1945
0.741775 0.73998
0.863353711056 0.838933987958 0.826070460705
0.842529 0.84481
0.899641664191 0.858035389746 0.851425474255
0.843386 0.843782
0.924207174519 0.881377450232 0.869913279133
SAVED

0.826936 0.824255
0.948435357 0.888215183139 0.887149051491
SAVED

0.863263 0.849435
0.960992416126 0.894910682772 0.892845528455
SAVED

0.919637 0.906475
0.967758480464 0.894241659599 0.892216802168
SAVED

0.939513 0.928058
0.974439012386 0.903987272756 0.895582655827
SAVED

0.916895 0.912641
0.977201259277 0.901369126951 0.893035230352
SAVED

0.933687 0.916752
0.982818788321 0.904550937949 0.902444444444
SAVED

0.939685 0.919322
0.98871422269 0.908596684384 0.892899728997
SAVED

0.95562 0.934738
0.991011893508 0.911488760937 0.883674796748
SAVED

0.922036 0.903392
0.992118112765 0.907148012158 0.88674796748
SAVED

0.964188 0.940904
0.99547158373 0.923341533696 0.902970189702
SAVED

0.952365 0.922919
0.997185833431 0.92422127283 0.909317073171
SAVED

0.966244 0.936793
0.99734849508 0.911446617745 0.908639566396
0.977382 0.954265
0.998068167836 0.919438020534 0.907241192412
SAVED

0.957676 0.926002
0.998120387627 0.906392068651 0.913149051491
0.969671 0.940904
0.998587964839 0.903212891602 0.909089430894
0.972755 0.941418
0.999113163889 0.917278181943 0.903165311653
0.982694 0.953751
0.999263220761 0.90983464065 0.911810298103
0.977724 0.951182
0.99907715024 0.92131339258 0.903658536585
SAVED

0.964702 0.937307
0.999022829652 0.912974308457 0.900216802168
0.978581 0.949126
0.999030332496 0.918489798714 0.899170731707
0.977039 0.945015
0.999322043054 0.926107180673 0.910124661247
SAVED

0.988862 0.961973
0.999493708115 0.929526047127 0.906601626016
SAVED

0.980809 0.957348
0.999418079452 0.921437188206 0.909116531165
0.983722 0.955293
0.999406074902 0.925174762549 0.904742547425
SAVED

0.981837 0.954779
0.999497909708 0.91332725769 0.899219512195
0.989205 0.964543
0.99971459183 0.92247233036 0.899994579946
0.991261 0.964029
0.999517116987 0.913416811973 0.898281842818
0.984236 0.959918
0.999300434865 0.918574085098 0.898417344173
0.98852 0.961459
0.999446890371 0.926196734956 0.906688346883
SAVED

0.992803 0.968654
0.999775214806 0.922625099432 0.905382113821
0.992803 0.969681
0.999867049612 0.920839281669 0.90620596206
0.83379 0.831963
0.871448378936 0.864214635277 0.829344173442
0.712646 0.700925
0.890151467406 0.875709190903 0.842802168022
0.833448 0.830935
0.917680901062 0.89169726438 0.856552845528
0.824366 0.813977
0.94693358783 0.904751118112 0.873398373984
0.879198 0.881295
0.947802117002 0.894573537236 0.873669376694
0.895305 0.89517
0.966223698782 0.915250040826 0.888054200542
0.896676 0.885406
0.965989610062 0.92453734677 0.892921409214
0.889479 0.880267
0.971599936376 0.91797881251 0.893111111111
0.933345 0.912127
0.984657885339 0.918342297541 0.895886178862
0.935744 0.915211
0.988516747847 0.923457427474 0.896113821138
0.931974 0.90853
0.991657738283 0.927213439464 0.890016260163
SAVED

0.934716 0.9111
0.990944067802 0.920607494113 0.899284552846
0.939856 0.908016
0.993488432116 0.930911504565 0.902742547425
SAVED

0.942769 0.916752
0.993253443055 0.929747298885 0.913501355014
SAVED

0.947567 0.922919
0.995657654251 0.928567289508 0.903766937669
0.960761 0.929599
0.996906127422 0.927139688878 0.912579945799
0.966244 0.934224
0.996768675328 0.921566251732 0.903322493225
0.959561 0.935252
0.99764920905 0.920533743527 0.909528455285
0.956991 0.927544
0.997817872974 0.930816682383 0.908189701897
SAVED

0.968986 0.94296
0.997536366283 0.933445363985 0.897669376694
SAVED

0.980466 0.958376
0.999149777766 0.940620242429 0.914482384824
SAVED

0.976696 0.952724
0.999256018031 0.937912542341 0.906639566396
SAVED

0.982351 0.95889
0.999507513348 0.927771836758 0.907322493225
0.979781 0.94964
0.999298634182 0.914818073108 0.908401084011
0.975326 0.945015
0.999498509935 0.926665577968 0.909848238482
0.984407 0.950154
0.999478102201 0.922625099432 0.914986449864
0.981494 0.955293
0.999503911983 0.923009656059 0.912357723577
0.982865 0.957862
0.999646165897 0.930253017189 0.914162601626
0.987834 0.961459
0.999655769537 0.922251078602 0.898048780488
0.982522 0.950154
0.999666573631 0.922904298079 0.88993495935
0.986635 0.957348
0.999666573631 0.921160623509 0.901517615176
0.985093 0.958376
0.999707989328 0.929078275711 0.910113821138
0.990404 0.961973
0.999713991603 0.920944639649 0.899658536585
0.990576 0.956321
0.999816030275 0.921139551913 0.903176151762
0.781528 0.782117
0.860105880129 0.843021877585 0.813761517615
0.813057 0.820658
0.885547722587 0.85499581202 0.829506775068
0.871145 0.872045
0.920340509053 0.872053268995 0.876021680217
0.881254 0.878212
0.937072450459 0.88979555284 0.873051490515
0.894277 0.893628
0.948231879882 0.890965026419 0.879170731707
0.882282 0.871531
0.958732859754 0.896180246432 0.87762601626
0.849897 0.836588
0.970410285498 0.908517665899 0.894173441734
0.904901 0.889003
0.977782279484 0.91306386274 0.888964769648
0.93146 0.907503
0.982084710105 0.906721312339 0.888672086721
0.920493 0.900822
0.984744918324 0.907111136865 0.900655826558
0.935401 0.914697
0.986471772802 0.908870615133 0.898048780488
0.939513 0.918808
0.989757418061 0.906299880419 0.899799457995
0.941398 0.918294
0.993812554958 0.909771425862 0.894124661247
0.936086 0.909044
0.993456019832 0.90613657555 0.893636856369
0.947567 0.923433
0.994868355107 0.9037554852 0.889398373984
0.946196 0.923947
0.996197858989 0.912921629467 0.906672086721
0.957505 0.931655
0.996816093299 0.906589614864 0.906146341463
0.957334 0.929085
0.997598189714 0.91533432721 0.894780487805
0.962303 0.93628
0.99831065974 0.909191956972 0.892964769648
0.969328 0.941932
0.998676198279 0.915982278788 0.904726287263
0.972755 0.940391
0.998978112705 0.91813684948 0.9127100271
0.976182 0.94296
0.999263220761 0.913206096013 0.910959349593
0.981494 0.947585
0.999250015756 0.913680206923 0.899555555556
0.981494 0.955807
0.999418379566 0.910914559946 0.894097560976
0.977896 0.943988
0.99918219005 0.914870752098 0.912260162602
0.978753 0.94964
0.999096357519 0.911646797908 0.902016260163
0.98475 0.951696
0.999518317442 0.912078765626 0.906178861789
0.986635 0.954265
0.999556131774 0.909292047053 0.904059620596
0.988005 0.954779
0.999577139736 0.921771699793 0.909035230352
0.981837 0.94964
0.999494608457 0.916477461294 0.915815718157
0.990576 0.957862
0.99970979001 0.916540676082 0.895013550136
0.988862 0.955807
0.999556131774 0.913898824732 0.896379403794
[[0.93344536398548161, 0.89766937669376701], [0.93791254234073829, 0.9066395663956639], [0.94062024242871223, 0.91448238482384825]]
0.906807588076
size of different sets: 10953 1946 1945
0.710761 0.691161
0.872659037775 0.842379193906 0.839479674797
0.80843 0.78777
0.904779611473 0.858988879465 0.853669376694
0.85658 0.848921
0.922884873367 0.877194738422 0.87308401084
SAVED

0.793523 0.782117
0.940835876797 0.89063841668 0.87945799458
SAVED

0.84647 0.827852
0.956112266549 0.894057283134 0.896043360434
SAVED

0.887594 0.863309
0.962282605107 0.883100053206 0.900970189702
SAVED

0.902844 0.877184
0.96781430162 0.89920928836 0.897100271003
SAVED

0.907814 0.878212
0.977002583979 0.893014239131 0.908368563686
SAVED

0.909184 0.881809
0.984001236469 0.90068956798 0.912325203252
SAVED

0.928033 0.898767
0.989381075428 0.90613657555 0.916715447154
SAVED

0.949452 0.922919
0.992697932516 0.9011109999 0.914634146341
SAVED

0.947738 0.916238
0.993958410237 0.899204020461 0.911598915989
0.941227 0.9111
0.995893543653 0.892787719474 0.909907859079
0.936943 0.904419
0.995217087276 0.895811493502 0.905403794038
0.951337 0.921891
0.995704471995 0.912405375364 0.903712737127
SAVED

0.964016 0.931141
0.997182232066 0.899767685654 0.90491598916
0.964873 0.93628
0.997712833164 0.895948458876 0.914552845528
0.96573 0.93371
0.998129090925 0.884917478362 0.909940379404
0.969842 0.938849
0.998145597181 0.897676329749 0.905344173442
0.979952 0.953751
0.998559153919 0.902196187095 0.91145799458
SAVED

0.975154 0.948613
0.999080751605 0.895342650491 0.91233604336
0.969328 0.930627
0.998257839721 0.905599249851 0.922130081301
SAVED

0.982694 0.949126
0.999027931586 0.9048143329 0.91487804878
0.968472 0.932682
0.998598168706 0.887598838955 0.908140921409
0.983722 0.95221
0.999136572761 0.891776282865 0.922796747967
0.988348 0.959918
0.998726617388 0.896654357343 0.916417344173
0.981837 0.948613
0.999063945235 0.906694972844 0.914514905149
SAVED

0.98475 0.951182
0.999321442827 0.90190118475 0.910525745257
0.981666 0.943474
0.999335848287 0.915239505028 0.912140921409
SAVED

0.988005 0.956321
0.999200196875 0.908839007739 0.906162601626
SAVED

0.981151 0.950668
0.999395871035 0.901853773659 0.911376693767
0.988005 0.959404
0.999753606617 0.897539364375 0.909886178862
0.992118 0.958376
0.999674376589 0.895974798371 0.904097560976
0.986463 0.951696
0.999539925632 0.904313882494 0.911815718157
0.850754 0.836588
0.859665913381 0.832633580749 0.825100271003
0.773989 0.761562
0.893704814125 0.863008286405 0.855105691057
0.834818 0.814491
0.922670892268 0.883468806136 0.868834688347
0.866861 0.848407
0.946053654335 0.891549763208 0.886677506775
0.903016 0.8926
0.950535552975 0.892055481512 0.885154471545
0.902502 0.899281
0.959644005078 0.908101501878 0.891474254743
0.94157 0.926002
0.972990063234 0.910993578431 0.892504065041
SAVED

0.914496 0.893114
0.979710810397 0.915639865353 0.900314363144
SAVED

0.92255 0.897225
0.982359614294 0.910740719279 0.896140921409
0.931289 0.9111
0.98829646436 0.913580116842 0.908075880759
SAVED

0.934716 0.911614
0.992199143475 0.906220861934 0.917506775068
0.928718 0.902878
0.993201523377 0.905936395387 0.9047100271
0.946367 0.91778
0.993415204362 0.915397541998 0.905918699187
SAVED

0.941912 0.901336
0.997484746719 0.908707310263 0.914975609756
0.956134 0.928571
0.997867091628 0.909139277982 0.907669376694
0.966587 0.931655
0.997859888898 0.904287542999 0.90287804878
0.97087 0.941418
0.998002142812 0.898002939488 0.910135501355
0.955278 0.921891
0.997924113239 0.89862455157 0.916509485095
0.962132 0.922405
0.998060965106 0.899857239937 0.908926829268
0.969671 0.937821
0.998072969655 0.904013612251 0.904363143631
0.980466 0.945015
0.998895281312 0.89090181163 0.908059620596
0.970699 0.936793
0.99883585879 0.888562864473 0.912563685637
0.972241 0.936793
0.998692404421 0.888468042291 0.906666666667
0.978924 0.947071
0.999277025993 0.892187178987 0.906227642276
0.977896 0.943988
0.999386267395 0.900252332362 0.910731707317
0.977039 0.941932
0.998685801919 0.895563902249 0.908411924119
0.980123 0.942446
0.998665094071 0.885623376829 0.908314363144
0.975668 0.940904
0.999098758429 0.902053953822 0.900016260163
0.980637 0.94964
0.999455893784 0.897892313609 0.900569105691
0.98355 0.954265
0.999546528134 0.895253096208 0.909322493225
0.978239 0.949126
0.999663572494 0.900710639576 0.914254742547
0.983722 0.950668
0.999562434163 0.897333916314 0.900932249322
0.98218 0.946557
0.999593946106 0.893151204505 0.909653116531
0.985949 0.957348
0.999689982503 0.901864309457 0.912802168022
0.777073 0.75591
0.871584030348 0.847989506345 0.81508401084
0.885024 0.87667
0.911094004628 0.877869029495 0.859886178862
0.873029 0.858684
0.927144687837 0.877273756908 0.862021680217
0.888622 0.869476
0.945767345824 0.893472546344 0.890379403794
0.931117 0.912641
0.966373755653 0.899119734076 0.904547425474
0.906614 0.884892
0.972152145663 0.899925722624 0.892666666667
0.923235 0.905961
0.97840891698 0.8974076669 0.91379403794
0.910727 0.889517
0.985944172842 0.901500824426 0.916943089431
0.933516 0.915725
0.989881064924 0.911404474553 0.911907859079
0.931117 0.910586
0.992613900668 0.917067465983 0.926330623306
SAVED

0.946367 0.92446
0.993041862866 0.917315057236 0.924937669377
SAVED

0.946881 0.919836
0.99407005255 0.914307086905 0.925029810298
0.95048 0.930627
0.996285492202 0.914581017653 0.92410298103
0.95425 0.936794
0.996993160408 0.918416048128 0.924861788618
SAVED

0.967101 0.947071
0.99767801997 0.90999267762 0.926823848238
0.971727 0.947071
0.99843970865 0.906415774197 0.93233604336
0.965216 0.937307
0.998465518431 0.904888083486 0.923468834688
0.971213 0.953751
0.998812449919 0.897792223527 0.920005420054
0.976868 0.951182
0.998916889501 0.913553777347 0.924216802168
0.986292 0.964543
0.999295032817 0.907832839029 0.923940379404
0.975154 0.951182
0.99941387786 0.904277007201 0.929821138211
0.971042 0.947071
0.999269223036 0.910235000975 0.925181571816
0.980295 0.951696
0.99938506694 0.907163815855 0.926216802168
0.976525 0.947585
0.999236210524 0.901485020729 0.913159891599
0.989034 0.959404
0.999360457613 0.91359065264 0.92274796748
0.981151 0.953237
0.999432184798 0.903339321179 0.929078590786
0.980809 0.956835
0.999490706978 0.914122710439 0.925951219512
0.985778 0.957348
0.99959754747 0.917704881762 0.930200542005
SAVED

0.986292 0.954779
0.999560933594 0.905504427669 0.928487804878
0.980123 0.954265
0.999624557707 0.903866111079 0.927327913279
0.991947 0.961973
0.999772813896 0.904092630736 0.928623306233
0.990404 0.964029
0.99971939365 0.90449299106 0.926390243902
[[0.9173150572357226, 0.92493766937669375], [0.91770488176200682, 0.93020054200542002], [0.91841604812752531, 0.92486178861788615]]
0.939479674797
size of different sets: 11022 1945 1945
0.741134 0.74653
0.864904341951 0.854925199996 0.818222222222
0.737365 0.744987
0.898101359601 0.888653599921 0.844428184282
0.789275 0.785604
0.923999561419 0.883800080983 0.851951219512
SAVED

0.889498 0.877121
0.93800985029 0.890590630027 0.855414634146
SAVED

0.875621 0.860154
0.953521982386 0.882694770019 0.852460704607
0.893952 0.878663
0.963979766011 0.903378311829 0.873111111111
SAVED

0.918109 0.90437
0.967347356068 0.902114317607 0.854623306233
SAVED

0.887442 0.866838
0.974579792089 0.899985773225 0.857609756098
SAVED

0.877848 0.861183
0.978614143641 0.914256322707 0.878601626016
SAVED

0.896522 0.876093
0.983910896957 0.904456263611 0.866959349593
SAVED

0.887956 0.861183
0.986159513057 0.905463081519 0.870579945799
SAVED

0.935755 0.916195
0.99048842502 0.912127778325 0.873685636856
SAVED

0.934041 0.910026
0.991702819957 0.909824135176 0.873468834688
SAVED

0.927874 0.901799
0.99283779619 0.899952942206 0.873826558266
0.955285 0.93162
0.994781776371 0.915810324261 0.895989159892
SAVED

0.948946 0.925964
0.995188945391 0.911629841207 0.887951219512
0.943978 0.914653
0.996782653533 0.909545071517 0.879972899729
0.958198 0.932648
0.997395481431 0.908198999748 0.885203252033
0.964023 0.936761
0.997523499638 0.905993849656 0.875658536585
0.965907 0.943959
0.997696561289 0.921780097836 0.896959349593
SAVED

0.961453 0.933676
0.99771908301 0.913539512131 0.895869918699
0.968306 0.941902
0.997937780781 0.910540945752 0.891105691057
0.969162 0.946015
0.998267901805 0.916554494019 0.896948509485
SAVED

0.976015 0.9491
0.998468819269 0.903569826105 0.878124661247
0.969162 0.94653
0.998361544753 0.905337229281 0.890585365854
0.977557 0.950129
0.998579649846 0.914294625562 0.895598915989
0.982354 0.955784
0.998903251425 0.913462906421 0.888471544715
0.980298 0.954242
0.998855837275 0.901567133962 0.874590785908
0.977728 0.950129
0.999063570523 0.926365496788 0.889170731707
SAVED

0.978756 0.9491
0.999367910103 0.914354815763 0.88389701897
0.980469 0.953727
0.99933412752 0.906147061077 0.880558265583
0.984752 0.956298
0.9992961962 0.912297405255 0.888173441734
0.983725 0.956812
0.999447328805 0.915405408363 0.893035230352
0.982525 0.950129
0.999264784325 0.914382174946 0.891842818428
0.794929 0.786632
0.876395754063 0.866492662267 0.825257452575
0.835532 0.824679
0.892792752747 0.857732252098 0.825046070461
0.877848 0.864781
0.924660988822 0.874262670037 0.855398373984
0.859517 0.852956
0.939771878667 0.908658634011 0.871333333333
0.881275 0.864781
0.955386543864 0.89837158147 0.868314363144
0.890526 0.873522
0.961073871247 0.912921194611 0.880498644986
0.906973 0.88946
0.972119590342 0.90054937238 0.878303523035
0.87802 0.862725
0.977393821936 0.911082657562 0.889490514905
0.918451 0.892031
0.983359707455 0.907443886317 0.877772357724
0.931472 0.908997
0.983606261039 0.906097814549 0.88145799458
0.933356 0.906427
0.99090329884 0.894114492706 0.884883468835
0.933185 0.912082
0.991642366914 0.901326373157 0.894758807588
0.959911 0.933162
0.993605312756 0.898645173293 0.887902439024
0.951174 0.928535
0.995958832664 0.903044529805 0.900298102981
0.960939 0.933676
0.996357704207 0.909862438031 0.910395663957
0.970019 0.936761
0.997949634318 0.905085524804 0.907886178862
0.971047 0.944473
0.998114991169 0.894606957987 0.898005420054
0.96368 0.928021
0.998340801062 0.899411230397 0.892769647696
0.97927 0.950643
0.998617581167 0.893507118859 0.89918699187
0.968477 0.93162
0.998587354646 0.895072064086 0.896124661247
0.974987 0.943959
0.998793606202 0.902130733117 0.905880758808
0.962138 0.92545
0.99857135237 0.902754522473 0.911073170732
0.98578 0.959897
0.999044901201 0.90124429561 0.90691598916
0.980127 0.950129
0.999418287638 0.900428991978 0.903073170732
0.972417 0.932648
0.999127875965 0.90309924817 0.89562601626
0.978756 0.947044
0.999041345139 0.90308283266 0.902894308943
0.986808 0.957841
0.999467479819 0.90737822428 0.903078590786
0.983039 0.952699
0.99902415751 0.904231918316 0.905447154472
0.990749 0.960925
0.999498891694 0.900543900544 0.906525745257
0.984239 0.95527
0.999335312874 0.906393293717 0.910341463415
0.988693 0.958869
0.999286120693 0.900122569137 0.915674796748
0.987322 0.959383
0.99957771772 0.903471333049 0.925566395664
0.992291 0.960411
0.999639948793 0.906070455366 0.917306233062
0.986466 0.94653
0.999619205102 0.899493307944 0.90952303523
0.777112 0.779949
0.86754293944 0.866875690819 0.816655826558
0.861573 0.861697
0.909397781018 0.87462928308 0.832775067751
0.854377 0.839075
0.920205836682 0.880402070543 0.859669376694
0.862258 0.84473
0.935255680808 0.886826006544 0.860509485095
0.851807 0.834447
0.961143214442 0.895657550587 0.883398373984
0.828679 0.809255
0.967275642165 0.900357858104 0.87379403794
0.899092 0.873008
0.971455792231 0.908560140955 0.890498644986
0.936611 0.917224
0.981036414068 0.894557711459 0.882975609756
0.933185 0.914653
0.986870132641 0.899340096523 0.886233062331
0.897893 0.879177
0.990695269253 0.903318121628 0.885192411924
0.946377 0.920308
0.990966122589 0.926704750648 0.900672086721
SAVED

0.951516 0.92545
0.99414464872 0.911963623231 0.892807588076
0.966078 0.936761
0.995354894918 0.914289153726 0.889707317073
0.965564 0.939332
0.996632113604 0.916165993631 0.89881300813
0.967963 0.940874
0.99591378922 0.914710485133 0.90227100271
0.957855 0.928021
0.997823394142 0.925862087834 0.902509485095
SAVED

0.974987 0.945501
0.997896293399 0.914464252492 0.897750677507
0.971389 0.943445
0.998410144258 0.915848627116 0.888135501355
0.982525 0.954756
0.998983855482 0.913763857426 0.892964769648
0.983725 0.960411
0.998846354444 0.920948378695 0.894617886179
0.968134 0.94036
0.998558313479 0.917347910306 0.894769647696
0.979442 0.944987
0.998800718324 0.917873206606 0.912216802168
0.983553 0.956812
0.998536977111 0.914486139838 0.895165311653
0.987322 0.957841
0.999130839349 0.905791391707 0.899398373984
0.979955 0.951671
0.998704704669 0.904182671788 0.882726287263
0.97927 0.9491
0.998763972358 0.917741882531 0.886531165312
0.983039 0.954756
0.998875395612 0.916259014851 0.892249322493
0.987494 0.957841
0.999223296943 0.903821530582 0.890395663957
0.983039 0.954756
0.999251152757 0.917533952745 0.897994579946
0.98835 0.958869
0.999325237367 0.924942819309 0.893761517615
0.990063 0.965553
0.999482889418 0.920964794204 0.888260162602
0.986466 0.960411
0.9995320816 0.921232914191 0.905127371274
[[0.92586208783391888, 0.90250948509485096], [0.92636549678803204, 0.88917073170731709], [0.92670475064841262, 0.90067208672086718]]
0.936059620596
size of different sets: 11022 1945 1945
0.76529 0.768638
0.873873024904 0.839885857491 0.823127371274
0.773 0.770694
0.902518876759 0.863294373858 0.83847696477
0.799041 0.803599
0.91936927326 0.867130131215 0.846531165312
SAVED

0.825595 0.82365
0.937566824319 0.882210512492 0.863024390244
SAVED

0.837759 0.830848
0.955019380534 0.88358941528 0.867143631436
SAVED

0.874422 0.863239
0.961918139469 0.885707015989 0.877078590786
SAVED

0.925133 0.911054
0.971707976246 0.890390907997 0.883772357724
SAVED

0.876135 0.864267
0.981050341975 0.885964192302 0.881235772358
SAVED

0.88847 0.870951
0.98009435416 0.88653873513 0.886108401084
SAVED

0.934898 0.915167
0.989013548594 0.884826050319 0.879642276423
0.941408 0.918766
0.991866695115 0.895736892216 0.881360433604
SAVED

0.939352 0.913111
0.992133399713 0.888727469713 0.90247696477
SAVED

0.930615 0.904884
0.994411945995 0.886122875559 0.889517615176
0.971047 0.943959
0.997180636061 0.889460695799 0.893170731707
SAVED

0.944492 0.916709
0.996947714045 0.896382568918 0.892682926829
SAVED

0.963509 0.93419
0.995754655477 0.901613644571 0.898173441734
SAVED

0.95974 0.933676
0.997110107512 0.902111581689 0.897024390244
SAVED

0.977043 0.953213
0.998194113533 0.897974873327 0.902178861789
SAVED

0.96762 0.942931
0.998251603191 0.899780579358 0.906579945799
SAVED

0.971732 0.943959
0.998249232483 0.898346958206 0.901257452575
0.979955 0.954242
0.998850799521 0.893176072754 0.902471544715
0.979784 0.9491
0.998820573 0.894193834335 0.90464498645
0.965051 0.937275
0.998901177056 0.899151318165 0.902417344173
0.982183 0.956298
0.999084314214 0.892754741346 0.906726287263
0.975672 0.948072
0.998886656473 0.891660374055 0.89447696477
0.983211 0.951671
0.999026231879 0.902636877989 0.898238482385
SAVED

0.978414 0.948072
0.998920142717 0.89824846515 0.903403794038
0.989892 0.966581
0.999227149343 0.901225144183 0.908319783198
0.988522 0.963496
0.999419176653 0.906669621458 0.917826558266
SAVED

0.987665 0.966067
0.999277526878 0.904338619127 0.903279132791
SAVED

0.986808 0.96144
0.999345981058 0.904721647679 0.898422764228
SAVED

0.985609 0.963496
0.99926567334 0.900338706677 0.909062330623
0.987665 0.959383
0.999281082939 0.903583505696 0.902796747967
0.986466 0.957326
0.999244929649 0.887879335062 0.905842818428
0.760836 0.769666
0.874152175717 0.833166442321 0.832292682927
0.808292 0.810797
0.899201664237 0.856104380752 0.852758807588
0.826452 0.815938
0.92308061591 0.866539172877 0.870704607046
0.873223 0.862211
0.943908466982 0.880623679919 0.881685636856
0.85986 0.851928
0.956138354492 0.885066811123 0.882509485095
0.842556 0.832905
0.964037552007 0.889312956214 0.896314363144
0.925818 0.905913
0.973454595024 0.896371625245 0.898802168022
0.886243 0.871979
0.978894776146 0.899802466704 0.908016260163
0.89241 0.87455
0.984230053459 0.897509767228 0.909642276423
0.910057 0.886889
0.989321740573 0.900557580135 0.901642276423
0.93901 0.908997
0.98994820004 0.89265624829 0.907864498645
0.950831 0.930591
0.993754370992 0.890472985543 0.907707317073
0.956142 0.933676
0.994376681721 0.895140462042 0.909214092141
0.954771 0.926992
0.996395339189 0.892743797673 0.910189701897
0.960768 0.937789
0.996445716724 0.896437287282 0.910520325203
0.954943 0.928535
0.99702476204 0.898461866772 0.918737127371
0.969334 0.941388
0.998264049406 0.906937741445 0.91379403794
SAVED

0.965222 0.93419
0.997756717992 0.897843549252 0.92366395664
0.97019 0.9491
0.998161516305 0.890396379833 0.919208672087
0.966592 0.933162
0.99816803575 0.899818882213 0.924520325203
0.971732 0.946015
0.998237378946 0.901728553137 0.919111111111
0.974816 0.942416
0.998399179735 0.896442759119 0.926075880759
0.973788 0.943959
0.998908881856 0.891999627915 0.926742547425
0.969334 0.942416
0.999086684921 0.898122612911 0.920628726287
0.982183 0.950643
0.99904697557 0.885947776793 0.91837398374
0.980641 0.953213
0.999214110451 0.898002232509 0.926731707317
0.985438 0.956812
0.999540675415 0.895654814669 0.92691598916
0.988008 0.963496
0.99936465038 0.891304704685 0.916579945799
0.990235 0.960925
0.999495631971 0.90207875067 0.927067750678
0.984924 0.954242
0.999348944442 0.898352430043 0.919376693767
0.991948 0.965553
0.999029195263 0.892448318505 0.918731707317
0.987494 0.953727
0.999473702927 0.897219759896 0.921761517615
0.985952 0.957841
0.999322570321 0.895167821224 0.922655826558
0.989378 0.957326
0.999520524401 0.902210074745 0.928563685637
0.831592 0.824165
0.875284781243 0.840958337437 0.813761517615
0.83005 0.817995
0.90926235435 0.864071374635 0.8492899729
0.780538 0.784062
0.923925773147 0.874276349628 0.851588075881
0.871509 0.858098
0.939478207271 0.873373496613 0.857024390244
0.853178 0.834961
0.952119412539 0.88586022741 0.864753387534
0.865684 0.846787
0.965510354065 0.889827308841 0.877376693767
0.88333 0.865296
0.97489539253 0.890708274511 0.886677506775
0.906801 0.879692
0.978658890746 0.898160915767 0.894016260163
0.883502 0.862725
0.984841696004 0.903271611018 0.887951219512
0.934041 0.91054
0.98976565556 0.893077579697 0.905035230352
0.930958 0.907455
0.991383663454 0.893132298062 0.905311653117
0.943293 0.917738
0.994381423136 0.905706578242 0.915403794038
SAVED

0.94295 0.915681
0.995320223321 0.902976131849 0.915143631436
0.963509 0.928535
0.996673897325 0.907298882651 0.914926829268
SAVED

0.957684 0.929563
0.996129819945 0.906013001083 0.912325203252
0.95717 0.926478
0.997532093453 0.910833689003 0.915528455285
SAVED

0.975672 0.9491
0.998285385773 0.903545202841 0.921430894309
0.967106 0.932134
0.998163294335 0.902319511474 0.913203252033
0.970361 0.936761
0.998484525207 0.890598837782 0.921051490515
0.962995 0.932648
0.998665291656 0.888087264848 0.926330623306
0.975672 0.945501
0.998921920747 0.903621808551 0.928260162602
0.974302 0.941388
0.999093204367 0.897318252952 0.928357723577
0.978585 0.952185
0.998781159987 0.896836731344 0.92443902439
0.985609 0.959383
0.998910659886 0.898095253729 0.928466124661
0.987151 0.957326
0.999096760428 0.908836468696 0.925983739837
SAVED

0.976015 0.944473
0.99907927646 0.899906431597 0.935170731707
0.977386 0.947044
0.999445847113 0.908486271162 0.929132791328
SAVED

0.983039 0.951671
0.99939132084 0.905356380708 0.926021680217
0.986808 0.954756
0.999295307184 0.888530483601 0.923891598916
0.97276 0.936247
0.998914808625 0.882205040656 0.921723577236
0.970704 0.933162
0.999163140239 0.907063593683 0.924493224932
0.982697 0.944987
0.99939132084 0.897870908434 0.934455284553
[[0.9084862711623275, 0.92913279132791327], [0.90883646869562362, 0.92598373983739846], [0.91083368900270301, 0.91552845528455284]]
0.937913279133

