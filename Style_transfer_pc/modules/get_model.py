import tensorflow as tf
from tensorflow.keras import models

# from style_transfer import content_layers, style_layers


def get_model(content_layers, style_layers):
    '''
    Создает модель с доступом к промежуточным слоям.

    Эта функция загрузит модель VGG19 и получит доступ к промежуточным уровням.
    Эти слои затем будут использоваться для создания новой модели, которая будет принимать входное изображение.
    и вернет выходные данные этих промежуточных уровней из модели VGG.

    Returns:
    -----------
        модель: tensorflow.python.keras.engine.functional.Functional
        Модель keras, которая принимает входные данные изображения и выводит
        промежуточные слои стиля и содержимого.
    '''
    # Загружает модель VGG, натренированную на датасете imagenet
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet') # include_top=False убирает завершающий полносвязный слой сети, который выдает метки классификации
                                                                                   # weights='imagenet' указывает на то, что веса берем, обученные на imagenet
    vgg.trainable = False # запрет на обучение сети. Мы ее не обучаем
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers] # какие слои мы берем для стиля
    content_outputs = [vgg.get_layer(name).output for name in content_layers] # какие слои берем для контента
    model_outputs = style_outputs + content_outputs # выход модели - слои стиля + слои контента
    # Строим модель
    return models.Model(vgg.input, model_outputs)