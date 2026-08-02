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


# Размер памяти модели (128 чисел)
hidden_size = 128


def forward(inputs, targets):
    # вектор (74, 1)   текущая буква в виде нулей и одной единицы
    h = np.zeros((hidden_size, 1))

    xs = {}  # one-hot векторы на каждом шаге
    hs = {}  # состояния памяти на каждом шаге
    ps = {}  # вероятности на каждом шаге

    hs[-1] = h  # память до начала — нули
    loss = 0

    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        # новая память — h_prev это hs[t-1]
        hs[t] = np.tanh(Wx @ xs[t] + Wh @ hs[t-1] + bh)

        # оценки и вероятности
        y_raw = Wy @ hs[t] + by
        exp_y = np.exp(y_raw - np.max(y_raw))
        ps[t] = exp_y / np.sum(exp_y)

        # добавляем к loss — правильный ответ это targets[t]
        loss += -np.log(ps[t][targets[t]][0] + 1e-8)

    return loss, xs, hs, ps


def backward(inputs, targets, xs, hs, ps):
    # градиенты — такого же размера как сами матрицы
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    dh_next = np.zeros((hidden_size, 1))

    # идём назад — от шага 99 до шага 0
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWy += dy @ hs[t].T
        dby += dy
        dh = Wy.T @ dy + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh
        dWx += dh_raw @ xs[t].T
        dWh += dh_raw @ hs[t - 1].T
        dbh += dh_raw
        dh_next = Wh.T @ dh_raw

    return dWx, dWh, dWy, dbh, dby


def train_step(inputs, targets):
    global Wx, Wh, Wy, bh, by
    # 1. forward
    loss, xs, hs, ps = forward(inputs, targets)

    # 2. backward
    dWx, dWh, dWy, dbh, dby = backward(inputs, targets, xs, hs, ps)

    # 3. clip градиентов
    for dparam in [dWx, dWh, dWy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    # 4. обновить веса
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    Wy -= learning_rate * dWy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    # 5. вернуть loss
    return loss / len(inputs)


def generate(start_char, length=300):
    h = np.zeros((hidden_size, 1))
    x = ch2i[start_char]
    result = start_char

    for t in range(length):
        # 1. one-hot для x
        x_onehot = np.zeros((vocab_size, 1))
        x_onehot[x] = 1

        # 2. обновить память
        h = np.tanh(Wx @ x_onehot + Wh @ h + bh)

        # 3. получить вероятности
        y_raw = Wy @ h + by
        exp_y = np.exp(y_raw - np.max(y_raw))
        probs = exp_y / np.sum(exp_y)

        # 4. выбрать следующий символ по вероятностям
        probs_flat = probs.ravel()
        x = np.random.choice(range(vocab_size), p=probs_flat)

        # 5. добавить символ к результату
        result += i2ch[x]

    return result


Wx = np.random.randn(hidden_size, vocab_size) * 0.01
Wh = np.random.randn(hidden_size, hidden_size) * 0.01
Wy = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

learning_rate = 0.001
epochs = 10

best_loss = float('inf')
best_weights = None

# for epoch in range(epochs):
#     total_loss = 0
#
#     for i in range(len(all_inputs)):
#         loss = train_step(all_inputs[i], all_targets[i])
#         total_loss += loss
#
#         if i % 500 == 0:
#             print(f"Эпоха {epoch + 1}, шаг {i}, потеря: {loss:.4f}")
#
#         avg_loss = total_loss / len(all_inputs)
#
#         # сохраняем если лучше предыдущего
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             best_weights = (Wx.copy(), Wh.copy(), Wy.copy(),
#                             bh.copy(), by.copy())
#     print(f"=== Эпоха {epoch + 1} завершена, средняя потеря: {total_loss / len(all_inputs):.4f} ===")

Wx, Wh, Wy, bh, by = best_weights
# print("Лучшая потеря:", best_loss)


print("=== М ===")
print(generate('М', 300))

print("=== И ===")
print(generate('и', 300))

print("=== В ===")
print(generate('В', 300))

print("=== ? ===")
print(generate('?', 300))