import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
mpl.rcParams['figure.figsize'] = (10,10) # определяем размер отображаемой картинки
mpl.rcParams['axes.grid'] = False
from tensorflow.keras.preprocessing import image as kp_image

def load_img(path_to_img):
    '''
    Загрузка и предобработка изображения

    Parameters
    ----------
    path_to_img: str
        Путь к изображению

    Returns
    -------
    img: numpy.array, shape = [1, height, width, channels]
        массив, соответствующий изображению
    '''
    max_dim = 512  # максимальная размерность изображения
    img = Image.open(path_to_img)  # открываем изобр. библиотекой PIL
    long = max(img.size)  # возвращает ммаксимальный элемент (длину или ширину)
    scale = max_dim / long  # я так понимаю, что это типа нормализации, тут делим заданную макс размерномть на полученный максимальный эелемент
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)  # изменение изображения, где применяется наша нормализация. 3 параметр количество каналов, остается неизменным

    img = kp_image.img_to_array(img)  # Преобразует экземпляр изображения PIL в массив Numpy.

    # Нужно расширить массив изображений, чтобы он имел заданную размерность для объявления батча. Добаляем новую размерность по оси 0 (строка)
    img = np.expand_dims(img, axis=0)
    return img  # возвращает массив, соответствующий изображению


def imshow(img, title=None):
    '''
    Вывод изображения на экран

    Parameters
    ----------
    img: numpy.array, shape = [1, height, width, channels]
        Массив, соответсвующий изображению
    title: str
        Заголовок изображения. По умолчанию: None
    '''

    # Удаляем размерности (батча)
    out = np.squeeze(img, axis=0)  # удаляет оси с одним элементом (длинной 1), но не сами элементы массива. т.е. тут удаляем размерность, заданную для батча
    # Нормализация для вывода
    out = out.astype('uint8')  # тип uint8 - целые числа в диапазоне от 0 по 255 (числа размером 1 байт)
    plt.imshow(out)  # строим график
    if title is not None:  # Если есть заголовок, выводим его
        plt.title(title)
    plt.imshow(out)