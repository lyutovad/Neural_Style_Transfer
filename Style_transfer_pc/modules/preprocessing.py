import numpy as np
import tensorflow as tf
from utils.load_img import load_img


def load_and_process_img(path_to_img: str):
    """
    Загрузка и обработка изображения

    Parameters
    ----------
    path_to_img: str
        Путь к изображению

    Returns
    -------
    img: tf.Tensor, shape = [1, height, width, channels]
        Нормализованный массив, соответсвующий изображению, с каналами BGR,
        с каждым каналом, нормализованным на среднее значение = [103.939, 116.779, 123.68]
    """
    img = load_img(path_to_img)  # загрузка изображения подготовленной функцией load_img
    img = tf.keras.applications.vgg19.preprocess_input(img)  # обрабатывает массив в соответсвии с требованиями сети VGG
    return img
