"""
Retrain the YOLO model for your own dataset.
"""

import os
import sys
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_yolo3.bin"


from ..yolo3.utils import get_classes, get_anchors, create_model, create_tiny_model, data_generator_wrapper_hdf5


def _main():
    annotation_path = 'train.txt'
    log_dir = 'D:/Documents/3dsMax/renderoutput/yolo'
    classes_path = '../model_data/villeroy_classes.txt'
    anchors_path = '../model_data/yolo_anchors.txt'
    train_hdf5_file_path = 'G:/Documents/3dsMax/renderoutput/arnold2/data.arnold.26K.h5'
    val_hdf5_file_path = 'G:/Documents/3dsMax/renderoutput/arnold/data.arnold.h5'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2,
                                  weights_path='D:/Documents/keras-yolo3/model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2,
                             weights_path='../model_data/yolo_weights.h5')  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    # with open(annotation_path) as f:
    #    lines = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    num_val = 1123
    num_train = 231255

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 4
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator_wrapper_hdf5(train_hdf5_file_path, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper_hdf5(val_hdf5_file_path, batch_size, input_shape, anchors,
                                                        num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator_wrapper_hdf5(train_hdf5_file_path, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper_hdf5(val_hdf5_file_path, batch_size, input_shape, anchors,
                                                        num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.

if __name__ == '__main__':
    _main()
