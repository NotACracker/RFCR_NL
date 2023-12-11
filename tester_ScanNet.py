import os
from os import makedirs
from os.path import exists, join
from helper_ply import read_ply, write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time


def log_string(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.timeStap = time.strftime('test_%Y-%m-%d_%H-%M-%S', time.gmtime())
        self.test_log_path = join(dataset.experiment_dir , self.timeStap)
        makedirs(self.test_log_path) if not exists(self.test_log_path) else None

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        fname = 'log_' + str(dataset.val_split) + '_test.txt'
        log_file_path = os.path.join(self.test_log_path, fname)
        self.log_out = open(log_file_path, 'a')

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            log_string("Model restored from " + restore_snap, self.log_out)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.data.shape[0], model.config.num_classes], dtype=np.float16)
                           for l in dataset.input_trees['test']]

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with validation/test data
        self.sess.run(dataset.test_init_op)

        # Test saving path

        test_path = self.test_log_path
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]): 
                    probs = stacked_probs[j, :, :]  # probs (40960,20)
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0] 
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['test'])), self.log_out)

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['test'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 12 < new_min:
                # if True:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()

                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        points = self.load_evaluation_points_scannet(file_path)
                        points = points.astype(np.float16)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], model.config.num_classes], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :] # (120636,20)

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1) 

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]

                        # Save ascii preds
                        ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                        np.savetxt(ascii_name, preds, fmt='%d')
                        log_string(ascii_name + ' has saved', self.log_out)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    #self.sess.close()
                    #import sys
                    #sys.exit()

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_evaluation_points_scannet(file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """
        print(file_path)
        vertex_data = read_ply(file_path)  # mesh.ply

        return np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
