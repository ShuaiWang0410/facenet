from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import data_flow_ops
from six.moves import xrange  # @UnresolvedImport
from datetime import datetime

import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import facenet
import lfw

import celeba

feature_name = "Wearing_Lipstick"
select_feature_image_path = "/home/ec2-user/input_imgs"

train_images = []
train_labels = []
train_size = -1
cur_start = -1
batch_size = -1

def load_pretrain_checkpoint(sess, checkpoint_filename):
    facenet_reader = tf.train.NewCheckpointReader(checkpoint_filename)
    #read checkpoint file from pre-trained model check-point
    var_to_shape_map = facenet_reader.get_variable_to_shape_map()
    # get the key of all variables saved in the checkpoint
    restored_tensor_list = []
    g = tf.get_default_graph()
    for key in var_to_shape_map:
        restored_tensor_list.append(g.get_tensor_by_name(key))
    saver = tf.train.Saver(restored_tensor_list)
    saver.restore(sess, checkpoint_filename)

def next_batch():

    global train_images, train_labels
    global cur_start, batch_size

    start = cur_start
    if start + batch_size < train_size:
        end = start + batch_size
    else:
        end = train_size

    image_batch = []
    for i in range(start, end):
        image_batch.append(train_images[i])
    label_batch = train_labels[start:end,:]

    if (end == train_size):
        cur_start = 0
    else:
        cur_start = end

    return np.ndarray(buffer=image_batch), label_batch

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def main(args):
    print("<------------ Start Running ----------------->")
    global train_images, train_labels
    global train_size, batch_size

    # Create a path for all checkpoints of logs and models in training
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    try:
        os.makedirs(log_dir)
        print("Model directory:" + model_dir)
    except FileExistsError:
        print("Model directory alreadly exist:" + model_dir)

    try:
        os.makedirs(model_dir)
        print("Log directory alreadly exist:" + log_dir)
    except FileExistsError:
        print("Log directory alreadly exist:" + log_dir)
    
    # write current hyperparameters to a file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    train_filenames, train_labels = celeba.getTrainingData(feature_name, select_feature_image_path)
    train_size = len(train_filenames)
    print("Training set size is " + str(train_size))
    train_labels = tf.one_hot(train_labels, depth=2)
    train_labels = tf.reshape(train_labels, shape=(train_size, 2))
    train_labels = tf.Session().run(train_labels);
    batch_size = args.batch_size

    with tf.Graph().as_default():

        # global_step of training
        g_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # set the random seed
        tf.set_random_seed(args.seed)

        mt_network = importlib.import_module(args.model_def)

        image_size = tf.Variable(args.image_size, trainable=False)

        image_batch_ph = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), name='image_batch')
        label_batch_ph = tf.placeholder(tf.float32, shape=(None, 2), name='label_batch')

        learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_ph = tf.placeholder(tf.bool, name='phase_train')

        learning_rate = tf.train.exponential_decay(learning_rate_ph, g_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        prelogits, feature1, _ = mt_network.inference(image_batch_ph, args.keep_probability,
                                                      phase_train=phase_train_ph,
                                                      bottleneck_layer_size=args.embedding_size,
                                                      weight_decay=args.weight_decay)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_batch_ph, logits=feature1)
        cross_entropy = tf.reduce_mean(loss)
        train_op = tf.train.AdagradDAOptimizer(learning_rate, global_step=g_step).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(feature1, 1), tf.argmax(label_batch_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(train_size):
            filename = train_filenames[i]

            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)
            if args.random_crop:
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)

            image.set_shape((args.image_size, args.image_size, 3))
            train_images.append(tf.image.per_image_standardization(image))
            print("No." + str(i) + " finished")
        print("<----------------Finish loading all training images---------------->")

        # ------------------------ SYSTEM CONFIGURATION ------------------------------------

        # Create saver to save all parameters of at most 3 check points during training
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Set gpu options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_ph: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_ph: True})

        # Set up writer and coordinator
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if args.pretrained_model:
                load_pretrain_checkpoint(sess, args.pretrained_model)
                print("<----------------Finish loading pretrained model---------------->")

            epoch = 0
            step = 0
            print("<----------------Start training---------------->")
            while epoch < args.max_nrof_epochs:

                step = 0

                while step < args.epoch_size:

                    image_batch_, label_batch_ = next_batch();
                    sess.run(train_op, feed_dict={image_batch_ph: image_batch_, label_batch_ph: label_batch_, learning_rate_ph:args.learning_rate})

                    print("Now running accuracy evaluation")
                    if step % 10 == 0:
                        accuracy = sess.run(accuracy, feed_dict={image_batch_ph: image, label_batch_ph: label_batch_})
                    print("Step %d of Epoch %d, the accuracy is %g" % (step, epoch, accuracy))
                    step += 1

                epoch += 1
                print("<----------------No." + str(epoch) + " Finished---------------->")

                '''
                train(args, sess, train_set, epoch, image_paths_ph, labels_ph, labels_batch,
                    batch_size_ph, learning_rate_ph, phase_train_ph, enqueue_op, input_queue, global_step,
                    embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                    args.embedding_size, anchor, positive, negative, triplet_loss)
                '''
            save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

def parse_argument(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        # help='Directory where to write event logs.', default='~/logs/facenet') # ShuaiWang: use mine
                        help='Directory where to write event logs.', default='~/sw-facenet-logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        # help='Directory where to write trained models and checkpoints.', default='~/models/facenet') # ShuaiWang: use mine
                        help='Directory where to write trained models and checkpoints.',
                        default='~/sw-facenet-cp-models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.',
                        default=0.98)  # ShuaiWang set 1 to 2
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default="/Home/ec2-user/20180402-114759/model-20180402-114759.ckpt-275")  # ShuaiWang 10-20:add pretrained models
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        # default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160') # Shuai: use mine
                        default='/home/ec2-user/output_align')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1_mt')
    parser.add_argument('--max_nrof_epochs', type=int,
                        # help='Number of epochs to run.', default=500) # Shuai: shrink the max epoch
                        help='Number of epochs to run.', default=80)
    parser.add_argument('--batch_size', type=int,
                        # help='Number of images to process in a batch.', default=90) # Shuai: shrink the batch_size to 50
                        help='Number of images to process in a batch.', default=50)
    parser.add_argument('--image_size', type=int,
                        # help='Image size (height, width) in pixels.', default=160) # Shuai: use our size
                        help='Image size (height, width) in pixels.', default=182)
    '''parser.add_argument('--people_per_batch', type=int,
                        # help='Number of people per batch.', default=45) # Shuai: shrink the number of people to 25
                        help='Number of people per batch.', default=45)'''
    '''parser.add_argument('--images_per_person', type=int,
                        # help='Number of images per person.', default=40) # Shuai: use mine
                        help='Number of images per person', default=40)'''
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    '''parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)'''
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             # 'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1) # ShuaiWang use mine
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        # help='Number of epochs between learning rate decay.', default=100)
                        help='Number of epochs between learning rate decay.',
                        default=1)  # ShuaiWang use mine 10-20: use 80 epoches
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.96)  # ShuaiWang dont know
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='/home/ubuntu/ShuaiWang/sw-face-net/facenet/data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='/home/ubuntu/ShuaiWang/sw-face-net/facenet/data/pairs.txt')  # Shuai: use absolute path
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/home/ubuntu/lfw-align')  # ShuaiWang: use mine
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)



if __name__ == "__main__":
    main(parse_argument(sys.argv[1:]))