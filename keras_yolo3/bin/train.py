"""
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import h5py
import argparse
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras_yolo3.yolo3.utils import read_classes, create_model, create_tiny_model, data_generator_wrapper_hdf5, make_dir
from keras_yolo3.yolo3 import get_yolo3_anchors


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.weights is None:
        raise ValueError("missing weights file!")

    return parsed_args


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--annotations', help='Path to h5 file containing annotations for training.')
    parser.add_argument('--classes', help='Path to a h5 file containing class label mapping.')
    parser.add_argument('--val-annotations',
                        help='Path to h5 file containing annotations for validation (optional).')

    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--weights', help='Initialize the model with weights from a file.')

    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.',
                        action='store_true')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
                        default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    classes_path = args.classes
    train_hdf5_file_path = args.annotations
    val_hdf5_file_path = args.val_annotations
    class_names = read_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_yolo3_anchors()

    logging, checkpoint = None, None

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2,
                                  weights_path=args.weights)
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2,
                             weights_path=args.weights)  # make sure you know what you freeze
    if args.tensorboard_dir and not args.tensorboard_dir == "":
        make_dir(args.tensorboard_dir)
        logging = TensorBoard(log_dir=args.tensorboard_dir)
    if args.snapshot_path and not args.snapshot_path == "":
        make_dir(args.snapshot_path)
        checkpoint = ModelCheckpoint(
            os.path.join(args.snapshot_path, 'ep{epoch:03d}-loss{loss:.3f}.h5'),
            #monitor='val_loss',
            #save_weights_only=True,
            #save_best_only=True,
            #period=3
        )

    train_hdf5_dataset = h5py.File(train_hdf5_file_path, 'r')
    train_dataset_size = len(train_hdf5_dataset['images'])

    batch_size = args.batch_size

    if not val_hdf5_file_path is None and os.path.isfile(val_hdf5_file_path):
        val_hdf5_dataset = h5py.File(val_hdf5_file_path, 'r')
        val_dataset_size = len(val_hdf5_dataset['images'])
        val_gen = data_generator_wrapper_hdf5(val_hdf5_dataset, val_dataset_size, batch_size,
                                              input_shape, anchors,
                                              num_classes)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    else:
        val_gen = None
        val_dataset_size = 0
        reduce_lr = None
        early_stopping = None

    train_gen = data_generator_wrapper_hdf5(train_hdf5_dataset, train_dataset_size, batch_size, input_shape, anchors,
                                            num_classes)


    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_dataset_size, val_dataset_size,
                                                                                   batch_size))
        model.fit_generator(train_gen,
                            steps_per_epoch=max(1, train_dataset_size // batch_size),
                            validation_data=val_gen,
                            validation_steps=max(1, val_dataset_size // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        if args.tensorboard_dir:
            model.save_weights(os.path.join(args.tensorboard_dir, 'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_dataset_size, val_dataset_size,
                                                                                   batch_size))
        model.fit_generator(train_gen,
                            steps_per_epoch=max(1, train_dataset_size // batch_size),
                            validation_data=val_gen,
                            validation_steps=max(1, val_dataset_size // batch_size),
                            epochs=args.epochs - 50,
                            initial_epoch=50,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        if args.tensorboard_dir:
            model.save_weights(os.path.join(args.tensorboard_dir, 'trained_weights_final.h5'))

    # Further training if needed.


if __name__ == '__main__':
    main()
