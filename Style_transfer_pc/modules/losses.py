import tensorflow as tf


def get_content_loss(base_content, target):
    '''
    Вычисляет потери по контенту: среднее значение элементов по различным измерениям тензора от
    корня разницы между базовым контеном и таргетом

    Inputs:
    ----------
    base_content: tf.Tensor, shape = [1, height, width, channels]

    target: tf.Tensor, shape = [1, height, width, channels]

    Returns:
    ----------
        Ошибка: float
    '''
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    '''
    Строим матрицу Грамма
    Inputs:
    ----------
    input_tensor: tf.Tensor, shape = [1, height, width, channels]
        На вход поступает карта признаков

    Returns:
    ----------
        Матрица Грамма: tf.Tensor
    '''

    # Сначала делаем каналы изображений
    channels = int(input_tensor.shape[-1])  # берем последний элемент размерности (это количество каналов)
    a = tf.reshape(input_tensor, [-1,
                                  channels])  # меняем форму массива. 3 столбца, а количество строк подбирается исходя из формы начального массива
    n = tf.shape(a)[
        0]  # первый элемент размерности массива а, полученного на предыдущем шаге (т.е. это количество строк)
    gram = tf.matmul(a, a, transpose_a=True)  # матрица Грамма = матричное умножение а на транспонированную а
    return gram / tf.cast(n, tf.float32)  # матрица грамма деленная на количество строк, преобразованное в формат флоат (это усредняет величины)


def get_style_loss(base_style, gram_target):
    '''
    Вычисляет среднее значение квадратов разницы между картами стилей формируемого изображения и стилевого
    (Эта функция вызывается для каждого слоя)

    Inputs:
    ----------
    base_style: tf.Tensor, shape = [height, width, channels] высота, ширина, количество фильтров каждого слоя
        карта стилей формируемого изображения

    gram_target: tf.Tensor, shape = [height, width, channels] - высота, ширина, количество фильтров каждого слоя
        матрица Грама соответствующего слоя l стилевого изображения.

    Returns:
    ----------
    loss: float
        Значение функции потерь для стиля на слое (среднее квадрата разницы карты стиля и матрицы грама соответсвующего слоя)
    '''
    # Масштабирует потери на данном слое по размеру карты объектов и количеству фильтров
    height, width, channels = base_style.get_shape().as_list()  # получаем списком размерность карты признаков
    gram_style = gram_matrix(base_style)  # вычисляем матрицу Грамма для карты признаков стиля

    return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, num_style_layers,num_content_layers):
    ''' Вычисление общих потерь

    Inputs:
    ----------
    model:
        Используемая модель
    loss_weights:
        Веса каждого вклада каждой функции потерь.
       (вес стиля, вес контента и общий вес как сумма 2 ошибок)
    init_image:

    gram_style_features:

    content_features:

    Returns:
    ----------
    loss: float
        Общая ошибка по контенту и стилю (сумма ошибок по стилю и контенту)
    style_score: float
        Ошибка по стилю
    content_score :float
        Ошибка по контенту
    '''

    style_weight, content_weight = loss_weights  # вес стиля, вес контента

    # Скармливаем изображение модели
    # Это даст нам представление содержимого и стиля на желаемых уровнях.
    model_outputs = model(init_image)

    # получаем карты признаков на слоях
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Накапливает потери по стилю со всех слоев
    # Здесь мы одинаково взвешиваем каждый вклад каждого слоя потерь
    weight_per_style_layer = 1.0 / float(num_style_layers)  # вес потери для каждого слоя (веса равны для каждого слоя)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0],
                                                               target_style)  # вес умножаем на среднее квадрата разницы между матрицей Грамма на слое и целевыми признаками стиля ???

    # Накапливает потери по контенту со всех слоев
    weight_per_content_layer = 1.0 / float(
        num_content_layers)  # вес потери для каждого слоя (веса равны для каждого слоя)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0],
                                                                     target_content)  # вес, умноженный на среднее квадрата разницы между картой признаков контента на слое и целевыми признаками контента ???

    style_score *= style_weight  # ошибка по стилю, умноженная на заданный вес
    content_score *= content_weight  # ошибка по контенту, умноженная на заданный вес

    # Вычисляем финальную ошибку
    loss = style_score + content_score
    return loss, style_score, content_score