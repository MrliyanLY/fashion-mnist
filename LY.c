wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp27-none-linux_x86_64.whl
pip install tensorflow_gpu-1.2.0-cp27-none-linux_x86_64.whl
cd $WORKSPACE
git clone https://github.com/tensorflow/models/
cd $WORKSPACE/data
wget http://download.tensorflow.org/example_images/flower_photos.tgz
tar zxf flower_photos.tgz
flower_photos
├── daisy
│   ├── 100080576_f52e8ee070_n.jpg
│   └── ...
├── dandelion
├── LICENSE.txt
├── roses
├── sunflowers
└── tulips
import os

class_names_to_ids = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
data_dir = 'flower_photos/'
output_path = 'list.txt'

fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))

fd.close()
daisy
dandelion
roses
sunflowers
tulips
import random

_NUM_VALIDATION = 350
_RANDOM_SEED = 0
list_path = 'list.txt'
train_list_path = 'list_train.txt'
val_list_path = 'list_val.txt'

fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)

fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)

fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)

fd.close()
import sys
sys.path.insert(0, '../models/slim/')
from datasets import dataset_utils
import math
import os
import tensorflow as tf

def convert_dataset(list_path, data_dir, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines = [line.split() for line in fd]
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_path = os.path.join(output_dir,
                    'data_{:05}-of-{:05}.tfrecord'.format(shard_id, _NUM_SHARDS))
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                        i + 1, len(lines), shard_id))
                    sys.stdout.flush()
                     image_data = tf.gfile.FastGFile(os.path.join(data_dir, lines[i][0]), 'rb').read()
                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(lines[i][1]))
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

os.system('mkdir -p train')
convert_dataset('list_train.txt', 'flower_photos', 'train/')
os.system('mkdir -p val')
convert_dataset('list_val.txt', 'flower_photos', 'val/')
data
├── flower_photos
├── labels.txt
├── list_train.txt
├── list.txt
├── list_val.txt
├── train
│   ├── data_00000-of-00005.tfrecord
│   ├── ...
│   └── data_00004-of-00005.tfrecord
└── val
    ├── data_00000-of-00005.tfrecord
    ├── ...
    └── data_00004-of-00005.tfrecord
    import os
import tensorflow as tf
slim = tf.contrib.slim

def get_dataset(dataset_dir, num_samples, num_classes, labels_to_names_path=None, file_pattern='*.tfrecord'):
    file_pattern = os.path.join(dataset_dir, file_pattern)
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    items_to_descriptions = {
        'image': 'A color image of varying size.',
        'label': 'A single integer between 0 and ' + str(num_classes - 1),
    }
    labels_to_names = None
    if labels_to_names_path is not None:
        fd = open(labels_to_names_path)
        labels_to_names = {i : line.strip() for i, line in enumerate(fd)}
        fd.close()
    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
            cd $WORKSPACE/models/slim
CUDA_VISIBLE_DEVICES="0" python train_image_classifier.py \
    --train_dir=train_logs \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=../../data/flowers \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=../../checkpoints/inception_resnet_v2_2016_08_30.ckpt \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --max_number_of_steps=1000 \
    --batch_size=32 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=10 \
    --optimizer=rmsprop \
    --weight_decay=0.00004
    from datasets import dataset_factory
    from datasets import dataset_classification
    dataset = dataset_factory.get_dataset(
    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    dataset = dataset_classification.get_dataset(
    FLAGS.dataset_dir, FLAGS.num_samples, FLAGS.num_classes, FLAGS.labels_to_names_path)
    tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
    tf.app.flags.DEFINE_integer(
    'num_samples', 3320, 'Number of samples.')

tf.app.flags.DEFINE_integer(
    'num_classes', 5, 'Number of classes.')

tf.app.flags.DEFINE_string(
    'labels_to_names_path', None, 'Label names file path.')
    cd $WORKSPACE/models/slim
python train_image_classifier.py \
    --train_dir=train_logs \
    --dataset_dir=../../data/train \
    --num_samples=3320 \
    --num_classes=5 \
    --labels_to_names_path=../../data/labels.txt \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=../../checkpoints/inception_resnet_v2_2016_08_30.ckpt \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits
    tensorboard --logdir train_logs/
    python eval_image_classifier.py \
    --checkpoint_path=train_logs \
    --eval_dir=eval_logs \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=../../data/flowers \
    --model_name=inception_resnet_v2
    from datasets import dataset_factory
    from datasets import dataset_classification
    python eval_image_classifier.py \
    --checkpoint_path=train_logs \
    --eval_dir=eval_logs \
    --dataset_dir=../../data/val \
    --num_samples=350 \
    --num_classes=5 \
    --model_name=inception_resnet_v2
    from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf

from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'test_path', '', 'Test image path.')

tf.app.flags.DEFINE_integer(
    'num_classes', 5, 'Number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.test_list:
        raise ValueError('You must supply the test list with --test_list')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        test_image_size = FLAGS.test_image_size or network_fn.default_image_size

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.Graph().as_default()
        with tf.Session() as sess:
            image = open(FLAGS.test_path, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)
            processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
            processed_images = tf.expand_dims(processed_image, 0)
            logits, _ = network_fn(processed_images)
            predictions = tf.argmax(logits, 1)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            np_image, network_input, predictions = sess.run([image, processed_image, predictions])
            print('{} {}'.format(FLAGS.test_path, predictions[0]))

if __name__ == '__main__':
    tf.app.run()
    python test_image_classifier.py \
    --checkpoint_path=train_logs/ \
    --test_path=../../data/flower_photos/tulips/6948239566_0ac0a124ee_n.jpg \
    --num_classes=5 \
    --model_name=inception_resnet_v2
