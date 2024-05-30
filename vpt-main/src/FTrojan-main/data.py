from os import listdir
from model import get_model
import random
import numpy as np
from image import *
import random
import tensorflow as tf
from hashlib import md5
from PIL import Image


def poison(x_train, y_train, param):
    target_label = param["target_label"]
    num_images = int(param["poisoning_rate"] * y_train.shape[0])

    if param["clean_label"]:
        index = np.where(y_train == target_label)
        index = index[0]
        index = index[:num_images]

        x_train[index] = poison_frequency(x_train[index], y_train[index], param)
        return x_train

    else:
        index = np.where(y_train != target_label)
        index = index[0]
        index = index[:num_images]
        x_train[index] = poison_frequency(x_train[index], y_train[index], param)
        y_train[index] = target_label
        return x_train


def poison_frequency(x_train, y_train, param):
    if x_train.shape[0] == 0:
        return x_train

    if param["clean_label"]:
        model = get_model(param)
        model.load_weights("model/{}.hdf5".format(param["dataset"]))
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        for batch in range(0, x_train.shape[0], 100):
            with tf.GradientTape() as tape:
                images = tf.convert_to_tensor(x_train[batch:batch+100], dtype=tf.float32)
                tape.watch(images)
                prediction = model(images)
                y_target = keras.utils.to_categorical(y_train[batch:batch+100], param["label_dim"])
                y_target = tf.convert_to_tensor(y_target, dtype=tf.float32)
                loss = loss_object(y_target, prediction)
            gradient = tape.gradient(loss, images)
            gradient = np.array(gradient, dtype=np.float)

            for i in range(images.shape[0]):
                x_train[batch+i] = x_train[batch+i] + (param["degree"] / 255.) * gradient[i] / (1e-20 + np.sqrt(np.sum(gradient[i] * gradient[i])))

    x_train *= 255.
    if param["YUV"]:
        # transfer to YUV channel
        x_train = RGB2YUV(x_train)

    # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]

    x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)
    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


def impose(x_train, y_train, param):
    x_train = poison_frequency(x_train, y_train, param)
    return x_train


def digest(param):
    txt = ""
    txt += param["dataset"]
    txt += str(param["target_label"])
    txt += str(param["poisoning_rate"])
    txt += str(param["label_dim"])
    txt += "".join(str(param["channel_list"]))
    txt += str(param["window_size"])
    txt += str(param["magnitude"])
    txt += str(param["YUV"])
    txt += str(param["clean_label"])
    txt += "".join(str(param["pos_list"]))
    hash_md5 = md5()
    hash_md5.update(txt.encode("utf-8"))
    return hash_md5.hexdigest()
