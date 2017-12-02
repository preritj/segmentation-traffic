import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from itertools import compress


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    l2_reg = tf.contrib.layers.l2_regularizer(0.001)

    def kernel_init(stddev):
        return tf.truncated_normal_initializer(stddev=stddev)

    net = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=1,
                           padding='same', kernel_regularizer=l2_reg,
                           kernel_initializer=kernel_init(0.01))
    net = tf.layers.conv2d_transpose(net, num_classes, 4,
                                     strides=2, padding='same',
                                     kernel_regularizer=l2_reg,
                                     kernel_initializer=kernel_init(0.01))

    skip_layers = [vgg_layer4_out, vgg_layer3_out]
    k_sizes = [4, 16]
    stddevs = [0.001, 0.0001]

    for skip_layer, k_size, stddev in zip(skip_layers, k_sizes, stddevs):
        strides = k_size // 2
        conv1x1 = tf.layers.conv2d(skip_layer, num_classes, 1,
                                   strides=1, padding='same',
                                   kernel_regularizer=l2_reg,
                                   kernel_initializer=kernel_init(stddev))
        net = tf.add(conv1x1, net)
        net = tf.layers.conv2d_transpose(net, num_classes, k_size,
                                         strides=strides, padding='same',
                                         kernel_regularizer=l2_reg,
                                         kernel_initializer=kernel_init(0.01))

    return net


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate,
             num_classes, reg_losses=None):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :param reg_losses: list of all regularization losses
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = \
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    total_loss = cross_entropy_loss
    if reg_losses is not None:
        total_loss += tf.add_n(reg_losses)
    solver = tf.train.AdamOptimizer(learning_rate)
    train_op = solver.minimize(total_loss)
    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image, correct_label,
             keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    n_batch_display = 5

    for epoch in range(epochs):
        batch_idx = 0
        sum_loss = 0.
        for image, label in get_batches_fn(batch_size):
            feed_dict = {input_image: image,
                         correct_label: label,
                         learning_rate: 1e-4,
                         keep_prob: 0.5}
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict=feed_dict)
            batch_idx += 1
            sum_loss += loss
            if batch_idx % n_batch_display == 0:
                print("Loss after {} batches : {:3.3f}"
                      .format(batch_idx, sum_loss/n_batch_display))
                sum_loss = 0.
        print("{} epochs finished.".format(epoch+1))


tests.test_train_nn(train_nn)


def initialize_uninitialized_vars(sess):
    # https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var))
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    epochs = 30
    batch_size = 8

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'),
                                                   image_shape, augment=True)

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        input_image, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        net = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        logits, train_op, cross_entropy_loss = \
            optimize(net, correct_label, learning_rate, num_classes, reg_losses)
        initialize_uninitialized_vars(sess)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image, correct_label,
                 keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
