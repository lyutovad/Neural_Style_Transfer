import numpy as np

def deprocess_img(processed_img):
    '''
    Инверсия препроцессинга
     Parameters
    ----------
    processed_img: numpy.array, shape = [1, height, width, channels]
        Нормализованный массив, соответсвующий изображению, с каналами BGR,
        с каждым каналом, нормализованным на среднее значение = [103.939, 116.779, 123.68]

    Returns
    -------
    x: numpy.array, shape = [1, height, width, channels]
        Массив, соответсвующий изображению
    '''

    x = processed_img.copy()  # создание копии обработанного изображения
    if len(x.shape) == 4:  # проверяемразмерность. если длина размерности == 4
        x = np.squeeze(x, 0)  # сжимаем одну размерность (которая предназначена для батча)
    assert len(
        x.shape) == 3  # проверяем размерность массива. На данном этапе должно остаться 3 размерности height, width, channels

    if len(x.shape) != 3:  # Если длина размерностей не  равна 3
        raise ValueError(
            "Invalid input to deprocessing image")  # вызываем ошибку: Неправильный вход на депроцессинг изображения

    # инверсия шага предварительной обработки Для каждого канала прибавляем среднее значение, на когорое нормировали и переставляем каалы опять в RGB
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')  # ограниченичивает элементы массива 0 и 255, переводит в тип uint8
    return x