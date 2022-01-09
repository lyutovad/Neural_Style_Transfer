import numpy as np
import tensorflow as tf
import time
from modules.deprocessing import deprocess_img
from modules.features import get_feature_representations
from modules.losses import compute_loss, gram_matrix
from modules.get_model import get_model
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
from modules.preprocessing import load_and_process_img
from utils.visualise_image import show_results

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--content", type=str, required=True,
                help="path to input content image")
ap.add_argument("-s", "--style", type=str, required=True,
                help="path to input style image")
# ap.add_argument("-r", "--result", type=str, default="result.png",
# 	help="path to output result image")

args = vars(ap.parse_args())

content_path = args["content"]
style_path = args["style"]


def compute_grads(cfg):
    """Вычисление градиента

    Inputs:
    ----------
    cfg: dict
        Словарь параметров для передачи в функцию для вычисления ошибки
        (model, loss_weights, init_image, gram_style_features, content_features)

    Returns:
    ----------
    tape.gradient(total_loss, cfg['init_image']) : тип???

    all_loss: list
        Список ошибок [loss, style_score, content_score]
    """
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg) # вычисление ошибки функцией compute_loss. передаем словарь cfg (будет задан позже)
    # Вычислить градиенты по входному изображению
    total_loss = all_loss[0] # берем первый элемент вычисленной ошибки (потому что all_loss состоит из loss, style_score, content_score)
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# Слой контента, с которого мы будем вытаскивать карты признаков
content_layers = ['block5_conv2']

# Слои стиля, которые нам будут нужны
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
               ]

num_content_layers = len(content_layers) # количество слоев контента
num_style_layers = len(style_layers) # количество слоев стиля


def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):

    """
    Процесс переноса стиля

    Inputs:
    ----------
    content_path: str
        Путь к изображению с контентом
    style_path: str
        Путь к изображению со стилем
    num_iterations: int
        Количество итераций. По умолчанию 1000
    content_weight=1e3:
    style_weight=1e-2:

    Returns:
    ----------
    best_img: tf.Tensor, shape = [batch, height, width, channels]
        Лучшее изображение
    best_loss: float
        Лучшая ошибка (минимальная)
    """

    model = get_model(content_layers, style_layers)  # загружаем модель
    for layer in model.layers:
        layer.trainable = False  # Нам не нужно обучать какие-либо слои модели, поэтому мы устанавливаем для них значение false.

    # Получаем представления карт стиля и контента (из указанных промежуточных слоев)
    style_features, content_features = get_feature_representations(model, content_path, style_path,content_layers, style_layers)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]  # строим матрицу грамма на каждом слое и получаем список матриц

    # Инициируем исходное изображение
    init_image = load_and_process_img(content_path)  # загружаем и обрабатываем функцией изображение контента (передаем путь)
    init_image = tf.Variable(init_image, dtype=tf.float32)  # объявляем переменной (для того, чтобы менять значения)
    # Задаем оптимизатор
    opt = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99,
                                   epsilon=1e-1)  # передаем в оптимизатор learning_rate(скорость обучения)
    # beta1 (экспоненциальная скорость убывания оценки момента первого порядка)
    # epsilon (очень маленькое число, чтобы предотвратить деление на ноль)

    # Для отображения промежуточных изображений
    iter_count = 1

    # Сохраняем лучший результат
    best_loss, best_img = float('inf'), None

    # Создаем конфигурацию
    loss_weights = (style_weight, content_weight)  # кортеж ошибок
    # Словарь для передачи в градинтную ленту в качестве параметров
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'num_style_layers': num_style_layers,
        'num_content_layers': num_content_layers
    }

    # Параметры для визуализации
    num_rows = 2  # количество строк
    num_cols = 5  # количество столбцов
    display_interval = num_iterations / (
                num_rows * num_cols)  # интервал вывода как количество итераций деленное на произведение количества строк и столбцов
    start_time = time.time()  # засекаем стартовое время (время в секундах с начала эпохи)
    global_start = time.time()  # засекаем стартовое время всей программы

    norm_means = np.array([103.939, 116.779, 123.68])  # среднее по каналам модели VGG19
    min_vals = -norm_means  # минимальные значения
    max_vals = 255 - norm_means  # максимальные значения

    imgs = []  # список тензоров, соответсвующих изображениям
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)  # вычисляем градиент и ошибки
        loss, style_score, content_score = all_loss  # приравниваем соответсвующим переменным ранее найденные ошибки
        opt.apply_gradients([(grads, init_image)])  # приминяем найденные градиенты для исходного изображения
        clipped = tf.clip_by_value(init_image, min_vals,
                                   max_vals)  # отсекаем значения за пределами минимального и максимального
        init_image.assign(
            clipped)  # заменыяет в (копии???) изображении старые значения на новые, полученные на предыдущем шаге
        end_time = time.time()  # записываем конечное время операции

        # Обновляет лучшую ошибку и лучшее изображение
        if loss < best_loss:  # если ошибка на этом шаге меньше лучшей ошибки
            best_loss = loss  # то заменяем  лучшую ошибку
            best_img = deprocess_img(init_image.numpy())  # то заменяем лучшее изображение

        if i % display_interval == 0:  # если остаток от деления номера итерации на интервал вывода равен нулю
            start_time = time.time()  # обновляем время старта эпохи

        # Используем метод .numpy(), чтобы получить конкретный массив numpy
        plot_img = init_image.numpy()  # переводим тензор изображения в нампай массив
        plot_img = deprocess_img(plot_img)  # проводим депроцессинг
        imgs.append(plot_img)  # добавляем в список изображений полученный массив
        IPython.display.clear_output(wait=True)  # очищаем окно вывода
        IPython.display.display_png(Image.fromarray(plot_img))  # строим изображение
        print('Iteration: {}'.format(i))  # печатает номер итерации
        print('Total loss: {:.4e}, '  # финальная ошибка на итерации
              'style loss: {:.4e}, '  # ошибка по стилю на итерации
              'content loss: {:.4e}, '  # ошибка по контенту на итерации
              'time: {:.4f}s'.format(loss, style_score, content_score,
                                     time.time() - start_time))  # время выполнения на итерации
    print('Total time: {:.4f}s'.format(time.time() - global_start))  # время выполнения всей функции
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14, 4))
    img_to_show = np.linspace(1, num_iterations, num_rows * num_cols, dtype=int)
    for i, img_index in enumerate(img_to_show):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(imgs[img_index - 1])
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss

if __name__ == '__main__':
    best, best_loss = run_style_transfer(content_path, style_path, num_iterations=10)
    show_results(best, content_path, style_path)