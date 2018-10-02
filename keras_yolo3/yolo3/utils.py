"""Miscellaneous utility functions."""
import h5py

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from PIL import Image
import numpy as np
import os
import sys
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from tqdm import tqdm, trange

from .model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss




def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data_bin(image, box, input_shape, random=True, max_boxes=20, jitter=0.3, hue=0.1, sat=1.5, val=1.5,
                        proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    image = Image.fromarray(image)
    iw, ih = image.size
    h, w = input_shape

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    return get_random_data_bin(image, box, input_shape, random, max_boxes, jitter, hue, sat, val, proc_img)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for
        l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_hdf5(hdf5_dataset, batch_size, input_shape, anchors, num_classes):
    n = len(hdf5_dataset['images'])
    if hdf5_dataset.attrs['has_labels']:
        labels = []
        l = hdf5_dataset['labels']
        label_shapes = hdf5_dataset['label_shapes']
        tr = trange(n, desc='Loading labels', file=sys.stdout)
        for i in tr:
            labels.append(l[i].reshape(label_shapes[i]))
    else:
        raise ValueError("wrong hdf5 format...")
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            image = hdf5_dataset['images'][i].reshape(hdf5_dataset['image_shapes'][i])
            box = load_annotations(i, labels)
            image, box = get_random_data_bin(image, box, input_shape, random=True)

            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def data_generator_wrapper_hdf5(hdf5_file, batch_size, input_shape, anchors, num_classes):
    hdf5_dataset = h5py.File(hdf5_file, 'r')
    dataset_size = len(hdf5_dataset['images'])
    if dataset_size == 0 or batch_size <= 0: return None
    return data_generator_hdf5(hdf5_dataset, batch_size, input_shape, anchors, num_classes)


def load_annotations(image_index, labels):
    annots = labels[image_index]
    boxes = np.zeros((len(annots), 5))

    for idx, annot in enumerate(annots):
        boxes[idx, 0] = float(annot[1])
        boxes[idx, 1] = float(annot[2])
        boxes[idx, 2] = float(annot[3])
        boxes[idx, 3] = float(annot[4])
        boxes[idx, 4] = annot[0]

    return boxes
