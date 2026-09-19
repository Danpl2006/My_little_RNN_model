"""
Данный файл содержит:
 - основные переменные для обучения модели
 - пути для файлов
 - алфавит нужных символов
"""
import os


# Универсальны пути
CONFIG_FILE = os.path.abspath(__file__)
RNN_FOLDER = os.path.dirname(CONFIG_FILE)
MAIN_FOLDER = os.path.dirname(RNN_FOLDER)

FOLDER_DATASETS = os.path.join(MAIN_FOLDER, 'datasets')
FOLDER_WEIGHTS  = os.path.join(MAIN_FOLDER, 'weights')

DATASET_PATH = os.path.join(FOLDER_DATASETS, 'full_dataset.txt')

# Набор символов для чистки
ALLOWED_CHARS = 'ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁйцукенгшщзхъфывапролджэячсмитьбюё,.!?:;-— \n'

# Длина одного отрезка текста, на котором модель учится за один шаг
SEQ_LENGTH   = 100

# Размер памяти h, сколько чисел модель использует, чтобы запомнить суть прочитанного на данный момент
HIDDEN_SIZE  = 128

# Значение насколько сильно веса двигаются на каждом шаге обучения
LEARNING_RATE = 0.001

# Количество эпох, тоесть циклов полного прохождения по датасету
EPOCHS  = 10