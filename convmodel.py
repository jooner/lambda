
def shape_check(shape):
    # catch input shape error
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))



class TextCNN(object):
    """ Creates a text-processing CNN that reduces dimensionality of contexts """
    def __init__(self, input_size, output_size, embed_size, kernel_sizes, num_kernels, bath_size):
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.batch_size = batch_size
        self.input_x = tf.placeholder(tf.int32, [None, input_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = tf.constat(0.0)

    def linear(_input, scope=None):
        shape = _input.get_shape().as_list()
        shape_check(shape)
        input_size = shape[1]

        # linear FF model
        with tf.variable_scope(scope or "SimpleLinear"):
            W = tf.get_variable('W', [self.output_size, input_size], dtype=input_.dtype)
            b = tf.get_variable('b', [self.output_size], dtype=input_.dtype)

        return tf.matmul(_input, tf.transpose(W)) + b

    def run_convolution(_input):
        # create a convolution and maxpool layer for each filter size
        pre_input = tf.placeholder(tf.int32, shape=[self.batch_size, self.input_size, self.embed_size])

        pooled_layers = []
        for i, kernel_size in enumerate(self.kernel_sizes):
            with tf.name_scope("conv-maxpool-%d" %filter_size):
                # conv layer
                W = tf.get_variable('W', [filter_size, self.embed_size, 1, self.num_kernels])
                b = tf.get_variable('b', [self.num_kernels])
                expanded_input = tf.expand_dims(_input, -1)
                conv = tf.nn.conv2d(expanded_input, W, strides=[1,1,1,1],
                                    padding='VALID', name='conv')
                # nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # maxpool over layers
                pooled_layer = tf.nn.maxpool(h, ksize=[1, self.input_size - kernel_size + 1, 1, 1],
                                             strides=[1,1,1,1], padding='VALID', name='pool')
                pooled_layers.append(pooled_layer)
        # combine pooled layers
        total_num_kernels = self.num_kernels * len(self.kernel_sizes)
        h_pool = tf.concat(pooled_layers, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, total_num_kernels])
        # dropout
        with tf.name_scope("dropout"):
            output = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return output
