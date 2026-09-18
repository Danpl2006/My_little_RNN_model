import re
import random
from rnn.config import ALLOWED_CHARS, SEQ_LENGTH


def load_and_clean(path) :

    # Открытие датасета
    text = open(path, encoding='utf-8').read()

    # Очищенный датасет
    clean = ''.join(c for c in text if c in ALLOWED_CHARS)

    # Чистка пробелов и переносов
    # r' +', r'\n+' - ищем один или более пробелов или переносов подряд
    # ' ', '\n' - заменяем на один пробел или перенос
    # clean   — в тексте
    clean = re.sub(r' +', ' ', clean)
    clean = re.sub(r'\n+', '\n', clean)
    clean = clean.strip()
    return clean


def build_vocab(text):

    # Список всех уникальных символов в тексте
    chars = sorted(set(text))

    # Количество символов в алфавите
    vocab_size = len(chars)

    # Перевод символа в числовой индекс
    ch2i = {ch: i for i, ch in enumerate(chars)}

    # Перевод из  числового индекса в символ
    i2ch = {i: ch for i, ch in enumerate(chars)}

    return  vocab_size, chars, ch2i, i2ch


def make_sequences(text, ch2i):
    # Все куски входа
    all_inputs = []
    # Все куски ответа (сдвинуто на 1 символ)
    all_targets = []

    for i in range(0, len(text), SEQ_LENGTH):
        # Берем срез 0:99
        ask = text[i:i + SEQ_LENGTH]
        # Тот же срез только на 1 больше 1:100
        ans = text[i + 1: i + 1 + SEQ_LENGTH]

        # Для последнего куска, если он окажется меньше 100 символов
        if len(ask) < SEQ_LENGTH or len(ans) < SEQ_LENGTH:
            continue

        # Перевод кусков в индексы
        all_inputs.append([ch2i[c] for c in ask])
        all_targets.append([ch2i[c] for c in ans])

    return all_inputs, all_targets


def shuffle_pairs(all_inputs, all_targets):
    """
    Перемешивает all_inputs и all_targets одинаковым образом,
    чтобы пара (input[i], target[i]) не развалилась.
    """
    indices = list(range (0,  len(all_inputs)))
    random.shuffle(indices)

    shuffled_inputs = [all_inputs[i] for i in  indices]
    shuffled_targets = [all_targets[i] for i in  indices]

    return shuffled_inputs, shuffled_targets