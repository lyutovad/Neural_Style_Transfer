import tensorflow as tf
from modules.preprocessing import load_and_process_img
def get_feature_representations(model, content_path, style_path, content_layers, style_layers):
    '''Вспомогательная функция для вычисления контента и представлений стилевых функций.

    Эта функция загружает и предварительно обрабатывает как контент, так и изображения стилей, расположенных по указанному пути.
    Затем он будет передавать в НС, чтобы получить выходы промежуточных слоев.

    Inputs:
    ----------
    model:
        Модель, которую мы используем
    content_path: str
        Путь к изображению с контетом
    style_path: str
        Путь к изображению со стилем

    Returns:
    ----------
    style_features: карты признаков???

    content_features: карты признаков???
    '''
    num_content_layers = len(content_layers) # количество слоев контента
    num_style_layers = len(style_layers) # количество слоев стиля
    
    # Загружаем и обрабатываем изображения контента и стиля при помощи функции load_and_process_img, написанной заранее
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # пакетное вычисление содержимого и функций стиля
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Получаем параметры (карты признаков???) стиля и контента из модели
    style_features = [style_layer[0] for style_layer in style_outputs[
                                                        :num_style_layers]]  # список первых эелементов (это что, первая карта???) каждого слоя модели для изображения стиля от 0 слоя до слоя №(количество слоев стиля - 1)
    content_features = [content_layer[0] for content_layer in content_outputs[
                                                              num_style_layers:]]  # список первых эелементов (это что, первая карта???) каждого слоя модели для изображения контента от слоя №(количество слоев стиля) до последнего слоя
    return style_features, content_features