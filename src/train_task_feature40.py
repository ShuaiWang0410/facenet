from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
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
select_feature_image_path = "/home/project03/Wearing_Lipstick"
# select_feature_image_path = "/Volumes/PowerExtension/Wearing_Lipstick_align"
# select_feature_image_path = "/Volumes/PowerExtension/Wearing_Lipstick_align/Wearing_Lipstick"
all_labels_path = "/home/project03/labels_40"
# all_labels_path = "/Volumes/PowerExtension/celebA-labels"



train_fnames = ''
train_labels = ''
train_size = -1
cur_start = 0
batch_size = -1
image_size_o = -1


def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist = []

    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
    else:
        varlist.append(tensor_name)
    return varlist


def build_tensors_in_checkpoint_file(graph, loaded_tensors):
    full_var_list = list()
    tensor_name_list = list()
    variable_collections = [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

    # Loop all loaded tensors
    # for i, tensor_name in enumerate(loaded_tensors[0]):
    for tensor_name in loaded_tensors:
        # Extract tensor
        try:
            tensor_aux = graph.get_tensor_by_name(tensor_name + ":0")

        except:
            print('Not found: ' + tensor_name)
        if not ((tensor_name + ":0") in variable_collections):
            continue
        else:
            tensor_name_list.append(tensor_name)
            full_var_list.append(tensor_aux)

    return full_var_list


def toOneHot(labels, size):
    global train_size
    labels = tf.one_hot(labels, depth=2)
    labels = tf.reshape(labels, shape=(size, 2))
    return labels


def next_batch():
    global train_images, train_labels, train_fnames
    global cur_start, batch_size, image_size_o, train_size

    start = cur_start
    if start + batch_size < train_size:
        end = start + batch_size
    else:
        end = train_size

    image_batch = []
    for i in range(start, end):
        image_batch.append(train_fnames[i])
        # file_contents = tf.read_file(filename)
        # image = tf.image.decode_image(file_contents, channels=3)
        # image.set_shape((image_size_o, image_size_o, 3))
        # image_batch.append(tf.image.per_image_standardization(image))

    label_batch = train_labels[start:end]
    # label_batch = toOneHot(label_batch, end - start)
    # label_batch = tf.Session().run(label_batch)

    if (end == train_size):
        cur_start = 0
    else:
        cur_start = end

    image_batch = np.array(image_batch)
    image_batch.shape = (image_batch.shape[0], 1)
    label_batch = np.array(label_batch, dtype=np.float32)
    label_batch.shape = (label_batch.shape[0], 40)
    return image_batch, label_batch
    # return np.array(image_batch), label_batch
    # return np.ndarray(shape=(end - start + 1, image_size_o, image_size_o, 3), buffer=np.array(image_batch)), label_batch


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
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def main(args):
    print("<------------ Start Running ----------------->")
    global train_images, train_labels, train_fnames
    global train_size, batch_size, image_size_o

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
    # facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    '''
    fnames, labels = celeba.getData(feature_name, select_feature_image_path)
    ratios = [200, 1, 700]
    test_fnames, test_labels, val_fnames, val_labels, train_fnames, train_labels = celeba.splitData(fnames, labels,
                                                                                                    ratios,
                                                                                                    args.batch_size)
    '''
    fnames, labels = celeba.getData40(select_feature_image_path, all_labels_path)
    ratios = [200, 1, 1000]
    test_fnames, test_labels, val_fnames, val_labels, train_fnames, train_labels = celeba.splitData40(fnames, labels,
                                                                                                    ratios,
                                                                                                    args.batch_size)



    # Shuai: required by validation and evaluation
    '''
    train_fnames = np.array(train_fnames)
    train_fnames.shape = (train_fnames.shape[0], 1)
    train_labels = np.array(train_labels)
    train_labels.shape = (train_labels.shape[0], 1)
    '''
    val_fnames = np.array(val_fnames)
    val_fnames.shape = (val_fnames.shape[0], 1)
    val_labels = np.array(val_labels, dtype=np.float32)
    val_labels.shape = (val_labels.shape[0], 40)

    test_fnames = np.array(test_fnames)
    test_fnames.shape = (test_fnames.shape[0], 1)
    test_labels = np.array(test_labels, dtype=np.float32)
    test_labels.shape = (test_labels.shape[0], 40)

    # Shuai: validation conversion and evaluation end


    # test_labels = toOneHot(test_labels)
    # val_labels = toOneHot(val_labels)
    # train_labels = toOneHot(train_labels)

    # train_size = train_fnames.shape[0]
    train_size = len(train_fnames)
    print("Training set size is " + str(train_size))
    batch_size = args.batch_size
    image_size_o = args.image_size

    with tf.Graph().as_default():

        # global_step of training
        g_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # set the random seed
        tf.set_random_seed(args.seed)

        mt_network = importlib.import_module(args.model_def)

        # image_size = tf.Variable(args.image_size, trainable=False)
        image_path_ph = tf.placeholder(tf.string, shape=(None, 1))  # , name='image_batch_ph')
        labels_ph = tf.placeholder(tf.float32, shape=(None, 40))  # name='label_batch_ph')
        # image_batch_ph = tf.placeholder(tf.float32, shape=(None, 182, 182, 3))# , name = "network_input")

        learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_ph = tf.placeholder(tf.bool, name='phase_train')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string],
                                              shapes=[(1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_path_ph])

        nrof_preprocess_threads = 1
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            images = []
            filenames = input_queue.dequeue()
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images])
            '''
            if args.random_crop:
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)
            # pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size, 3))
            images.append(tf.image.per_image_standardization(image))
            '''

        image_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size,
            shapes=[(image_size_o, image_size_o, 3)], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        # labels_batch = toOneHot(labels_batch, batch_size)

        # Test queue

        learning_rate = tf.train.exponential_decay(learning_rate_ph, g_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size / 10,
                                                   args.learning_rate_decay_factor, staircase=True)

        # tf.summary.scalar('learning_rate', learning_rate)

        prelogits, features, _ = mt_network.inference(image_batch, args.keep_probability,
                                                      phase_train=phase_train_ph,
                                                      bottleneck_layer_size=args.embedding_size,
                                                      weight_decay=args.weight_decay)
        features_sig = tf.keras.backend.sigmoid(features)

        loss = tf.keras.backend.binary_crossentropy(target=labels_ph, output=features_sig)
        cross_entropy = tf.reduce_mean(loss)
        train_op = tf.train.AdagradDAOptimizer(learning_rate, global_step=g_step).minimize(cross_entropy)
        # train_op = tf.train.AdagradDAOptimizer(args.learning_rate, global_step=g_step).minimize(cross_entropy)


        # correct_prediction_val = tf.equal(tf.argmax(feature1, 1), tf.argmax(labels_batch, 1))
        # accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))
        #
        accuracy_val = tf.keras.metrics.binary_accuracy(y_true=labels_ph, y_pred=features_sig)
        #
        '''
        output_ph = tf.placeholder(tf.float32, shape=(None, 2))
        # test_labels_ph = tf.placeholder(tf.float32, shape=(None, 2))
        test_labels_one = toOneHot(test_labels, batch_size * ratios[0])
        correct_prediction = tf.equal(tf.argmax(output_ph, 1), tf.argmax(test_labels_one, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        '''
        #

        output_ph = tf.placeholder(tf.float32, shape=(None, 40))
        output_ph2 = tf.placeholder(tf.float32, shape=(40,None))
        labels_ph2 = tf.placeholder(tf.float32, shape=(40,None))
        accuracy_n = tf.keras.metrics.binary_accuracy(y_true=labels_ph2, y_pred=output_ph2)

        accuracy = tf.keras.metrics.binary_accuracy(y_true=test_labels, y_pred=output_ph)

        print("<----------------Finish loading all training images---------------->")

        # ------------------------ SYSTEM CONFIGURATION ------------------------------------

        # Create saver to save all parameters of at most 3 check points during training
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Set gpu options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tf.get_default_graph())

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_ph: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_ph: True})

        # Set up writer and coordinator
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        # ------------------------ START RUNNING GRAPH ------------------------------------

        with sess.as_default():

            if args.pretrained_model:
                # saver.restore(sess, os.path.expanduser(args.pretrained_model))

                restored_vars = get_tensors_in_checkpoint_file(file_name=args.pretrained_model)
                tensors_to_load = build_tensors_in_checkpoint_file(sess.graph, restored_vars)
                # tf.reset_default_graph()
                loader = tf.train.Saver(tensors_to_load)
                loader.restore(sess, args.pretrained_model)

            if args.finetune_model:

                print("<------ Now start finetuning training ------>",)
                saver.restore(sess, args.finetune_model)

                '''
                facenet_reader = tf.train.NewCheckpointReader(args.pretrained_model)
                # read checkpoint file from pre-trained model check-point
                var_to_shape_map = facenet_reader.get_variable_to_shape_map()
                # get the key of all variables saved in the checkpoint
                restored_tensor_list = []
                g = tf.get_default_graph()
                for key in var_to_shape_map:
                    cur_tensor = g.get_tensor_by_name(key)
                    restored_tensor_list.append(cur_tensor)
                saver = tf.train.Saver(restored_tensor_list)
                saver.restore(sess, args.pretrained_model)
                '''
            epoch = 0
            step = 0
            print("<----------------Start training---------------->")
            while epoch < args.max_nrof_epochs:

                l_step = 0
                '''
                while l_step < args.epoch_size:

                    image_batch_, label_batch_ = next_batch()
                    # print("Step %d of Epoch %d, batch size %s" % (l_step, epoch, image_batch_[0]))
                    # sess.run(enqueue_op, feed_dict={image_path_ph:train_fnames, label_ph: train_labels})
                    sess.run(enqueue_op, feed_dict={image_path_ph: image_batch_})
                    los, _= sess.run([loss, train_op],
                                           feed_dict={labels_ph: label_batch_, learning_rate_ph: args.learning_rate, phase_train_ph: True})

                    summary = tf.Summary()
                    los_v = tf.reduce_mean(los)
                    summary.value.add(tag='loss', simple_value=sess.run(los_v))

                    if step % 5 == 0:
                        print("Now running accuracy evaluation")
                        sess.run(enqueue_op, feed_dict={image_path_ph: val_fnames})
                        val_m = sess.run([accuracy_val], feed_dict={labels_ph: val_labels, phase_train_ph: False})
                        summary = tf.Summary()
                        val_m = np.mean(val_m)
                        summary.value.add(tag='val_m', simple_value=val_m)
                        print("Step %d of Epoch %d, the validation rate is %g" % (l_step, epoch, val_m))

                    summary_writer.add_summary(summary, step)
                    step += 1
                    l_step += 1

                '''

                epoch += 1
                print("<----------------No." + str(epoch) + " Epoch Finished---------------->")

                print("<----------------Start evaluating---------------->")

                # sess.run(enqueue_acc_op, feed_dict={image_path_ph: test_fnames, label_ph: test_labels})
                sess.run(enqueue_op, feed_dict={image_path_ph: test_fnames})
                # sess.run([feature1, test_image_batch])
                base = np.array([])
                base.shape = (0, 40)
                bn = 0
                for i in range(ratios[0]):
                    features_sig_ = sess.run(features_sig, feed_dict={phase_train_ph: False})
                    # print(str(bn) + " test batch out of queue")
                    base = np.concatenate((base, features_sig_))
                    bn += 1


                accuracy_m = sess.run([accuracy], feed_dict={output_ph: base})
                accuracy_n = sess.run([accuracy_n], feed_dict={output_ph2: np.transpose(base), labels_ph2: np.transpose(test_labels)})
                summary = tf.Summary()
                print(accuracy_n)
                accuracy_m = np.mean(accuracy_m)
                accuracy_n = np.mean(accuracy_n)
                summary.value.add(tag='accuracy_m', simple_value=accuracy_m)
                summary_writer.add_summary(summary, step)

                print("Epoch %d, the accuracy is %g" % (epoch, accuracy_n))
                print("<----------------End evaluating---------------->")

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
                        help='Directory where to write event logs.', default='~/sw-facenet-project2/logs')
    parser.add_argument('--models_base_dir', type=str,
                        # help='Directory where to write trained models and checkpoints.', default='~/models/facenet') # ShuaiWang: use mine
                        help='Directory where to write trained models and checkpoints.',
                        default='~/sw-facenet-project2/models')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.',
                        default=0.98)  # ShuaiWang set 1 to 2
    parser.add_argument('--finetune_model', type=str,
                        help='Load a pretrained model before training starts.',
                        # default="/Volumes/PowerExtension/20180402-114759/model-20180402-114759.ckpt-275")  # ShuaiWang 10-20:add pretrained models
                        default="/home/project03/20181202-203525/model-20181202-203525.ckpt-6000")  # ShuaiWang 10-20:add pretrained models
                        #)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.'#,
                        # default="/Volumes/PowerExtension/20180402-114759/model-20180402-114759.ckpt-275")  # ShuaiWang 10-20:add pretrained models
                        # default="/home/project03/20180402-114759/model-20180402-114759.ckpt-275")  # ShuaiWang 10-20:add pretrained models
                        )
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        # default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160') # Shuai: use mine
                        default='/home/project03/output_align')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1_mt_40')
    parser.add_argument('--max_nrof_epochs', type=int,
                        # help='Number of epochs to run.', default=500) # Shuai: shrink the max epoch
                        help='Number of epochs to run.', default=1)
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
                        help='Dimensionality of the embedding.', default=512)
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
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.05)
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
                        # default='/home/ubuntu/ShuaiWang/sw-face-net/facenet/data/learning_rate_schedule.txt')
                        )

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        # default='/home/ubuntu/ShuaiWang/sw-face-net/facenet/data/pairs.txt')  # Shuai: use absolute path
                        )
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        # default='/home/ubuntu/lfw-align')  # ShuaiWang: use mine
                        )
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_argument(sys.argv[1:]))