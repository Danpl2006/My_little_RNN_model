import numpy as np
import re

# Открытие датасета
text = open('datasets/master_dataset2.txt', encoding='utf-8').read()

# Нужные символы
allowed = 'ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁйцукенгшщзхъфывапролджэячсмитьбюё,.!?:;-— \n'

# Очищенный датасет
clean = ''.join(c for c in text if c in allowed)

# Чистка пробелов и переносов
clean = re.sub(r' +', ' ', clean)
clean = re.sub(r'\n+', '\n', clean)
clean = clean.strip()

# Множество
chars = sorted(set(clean))
vocab_size = len(chars)

# Словари символов и индексов
ch2i = {ch: i for i, ch in enumerate(chars)}
i2ch = {i: ch for i, ch in enumerate(chars)}

# Длина одного куска текста
seq_length = 100

# Все куски входа (список 11656 кусков текста)
all_inputs = []
# Все куски ответа (сдвинуто на 1 символ)
all_targets = []

for i in range(0, len(clean), seq_length):
    ask = clean[i:i + seq_length]
    ans = clean[i + 1: i+ 1 + seq_length]

    if len(ask) < seq_length or len(ans) < seq_length:
        continue

    all_inputs.append([ch2i[c] for c in ask])
    all_targets.append([ch2i[c] for c in ans])

# print("Кусков всего:", len(all_inputs))

# Размер памяти модели (128 чисел)
hidden_size = 128

# матрица (128, 74)  буква → память
Wx = np.random.randn(hidden_size, vocab_size) * 0.01

# матрица (128, 128) память → память
Wh = np.random.randn(hidden_size, hidden_size) * 0.01

# матрица (74, 128)  память → предсказание
Wy = np.random.randn(vocab_size, hidden_size) * 0.01

# вектор  (128, 1)   поправка для памяти
bh = np.zeros((hidden_size, 1))

# вектор  (74, 1)    поправка для предсказания
by = np.zeros((vocab_size, 1))


def forward(x, h_prev):
    # вектор (74, 1)   текущая буква в виде нулей и одной единицы
    x_onehot = np.zeros((vocab_size, 1))
    x_onehot[x] = 1
    h_new = np.tanh(Wx @ x_onehot + Wh @ h_prev + bh)
    y_raw = Wy @ h_new + by
    exp_y = np.exp(y_raw - np.max(y_raw))
    probs = exp_y / np.sum(exp_y)
    return h_new, probs

# вектор (128, 1)  память ДО текущего символа
h_prev = np.zeros((hidden_size, 1))  # память пустая — начало текста
x = ch2i[clean[0]]                   # индекс первого символа

# вектор (128, 1)  память ПОСЛЕ текущего символа
# probs    — вектор (74, 1)   вероятности для каждой буквы
h_new, probs = forward(x, h_prev)

target = ch2i[clean[1]]
# Насколько модель ошиблась
loss = -np.log(probs[target][0])
# print(loss)


def compute_loss(ask_list, ans_list):
    h = np.zeros((hidden_size, 1))  # начальная память
    total_loss = 0

    for t in range(len(ask_list)):
        x = ask_list[t]  # текущий символ
        target = ans_list[t]  # правильный следующий символ

        h, probs = forward(x, h)
        loss = -np.log(probs[target][0])
        total_loss += loss

    return total_loss / len(ask_list)


test_loss = compute_loss(all_inputs, all_targets)
print("Потеря на первом куске:", test_loss)

# Проверки
# print(len(clean))
# print(clean[:200])
# print(vocab_size)
# print(chars)
# print(ch2i['а'])
# print(i2ch[0])
# print(repr(i2ch[0]))
# print(Wx.shape, Wh.shape, Wy.shape)
# print(bh.shape, by.shape)
