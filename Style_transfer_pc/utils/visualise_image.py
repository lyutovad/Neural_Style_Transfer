import matplotlib.pyplot as plt
import numpy as np
from utils.load_img import load_img


def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))
        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()

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
    out = np.squeeze(img, axis=0) # удаляет оси с одним элементом (длинной 1), но не сами элементы массива. т.е. тут удаляем размерность, заданную для батча
    # Нормализация для вывода 
    out = out.astype('uint8') # тип uint8 - целые числа в диапазоне от 0 по 255 (числа размером 1 байт)
    plt.imshow(out) # строим график
    if title is not None: # Если есть заголовок, выводим его
        plt.title(title)
    plt.imshow(out)