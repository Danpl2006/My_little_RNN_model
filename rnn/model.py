import numpy as np
from rnn.config import HIDDEN_SIZE, LEARNING_RATE, FOLDER_WEIGHTS


class RNN:
    """Символьная рекуррентная нейронная сеть - которая умеет
    работать с последовательностями и обладает аналогом
    краткосрочной памяти"""

    def __init__(self, vocab_size):
        """
        W  — weight, вес. Все матрицы весов начинаются с W
        x  — input, вход. Wx работает с входным символом
        h  — hidden, скрытый. Wh работает со скрытым состоянием (памятью)
        y  — output, выход. Wy работает с выходом
        :param vocab_size:
        """
        self.vocab_size = vocab_size
        self.hidden_size = HIDDEN_SIZE

        # Матрица (128, 74) — переводит символ в пространство памяти
        self.Wx = np.random.randn(self.hidden_size, vocab_size) * 0.01

        # Матрица (128, 128) — обновляет память с учётом предыдущего состояния
        self.Wh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01

        # Матрица (74, 128) — переводит память в оценки для каждого символа
        self.Wy = np.random.randn(vocab_size, self.hidden_size) * 0.01

        # Вектор (128, 1) — смещение скрытого слоя, инициализируем нулями
        self.bh = np.zeros((self.hidden_size, 1))

        # Вектор (74, 1) — смещение выходного слоя, инициализируем нулями
        self.by = np.zeros((vocab_size, 1))

    def forward(self, inputs, targets):
        """
        Прямой проход по последовательности символов.

        Принимает:
            inputs  — список индексов входных символов (длина seq_length)
            targets — список индексов правильных следующих символов

        Возвращает:
            loss — суммарная потеря по всей последовательности
            xs   — one-hot векторы на каждом шаге
            hs   — состояния памяти на каждом шаге
            ps   — вероятности на каждом шаге
        """

        # вектор (74, 1)   текущая буква в виде нулей и одной единицы
        h = np.zeros((self.hidden_size, 1))

        xs = {}  # one-hot вектор символа на шаге t
        hs = {}  # память после шага t
        ps = {}  # вероятности после шага t

        hs[-1] = h  # память до начала — нули

        loss = 0

        for t in range(len(inputs)):
            # Пустой one-hot вектор
            xs[t] = np.zeros((self.vocab_size, 1))
            # Единица на позицию текущего символа
            xs[t][inputs[t]] = 1

            # Новая память, с учетом текущей буквы и прошлой памяти
            # tanh - сжимает сумму в диапазон от -1 до 1
            hs[t] = np.tanh(self.Wx @ xs[t] + self.Wh @ hs[t - 1] + self.bh)

            # Сырые оценки
            y_raw = self.Wy @ hs[t] + self.by

            # Процесс softmax приводит числа в вероятности
            # exp делает все числа положительными и вычитаем максимум, чтобы числа не стали слишком большими
            exp_y = np.exp(y_raw - np.max(y_raw))

            # Делим на сумму — получаем вероятности которые в сумме дают 1.0
            ps[t] = exp_y / np.sum(exp_y)

            # Считаем потерю для этого шага. targets[t] это индекс правильной следующей буквы.
            # Берём её вероятность из ps[t], берём логарифм с минусом.
            # Чем выше вероятность правильной буквы — тем меньше потеря.
            # 1e-8 защита от логарифма нуля.
            loss += -np.log(ps[t][targets[t]][0] + 1e-8)

        return loss, xs, hs, ps

    def backward(self, inputs, targets, xs, hs, ps):
        # Градиенты — такого же размера как сами матрицы
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Градиент от следующего шага. На самом последнем шаге (99) следующего шага нет — поэтому начинаем с нулей.
        dh_next = np.zeros((self.hidden_size, 1))

        # Идём назад — от шага 99 до шага 0
        for t in reversed(range(len(inputs))):
            # Копируем вероятности шага t
            dy = np.copy(ps[t])

            # Это производная softmax + cross-entropy вместе. На позиции правильной буквы вычитаем 1. Это говорит модели "ты дала слишком мало вероятности правильной букве".
            dy[targets[t]] -= 1

            # Градиент по Wy: ошибка выхода (dy) умноженная на память, которая её произвела (hs[t]). Суммируется по всем шагам, т.к. Wy общая для всех
            dWy += dy @ hs[t].T

            # Градиент по by — совпадает с dy, т.к. by прибавляется к y_raw напрямую (производная = 1).
            dby += dy

            # Общий градиент по hs[t]: вклад от выхода этого шага (через Wy) + вклад, пришедший "из будущего" от hs[t+1] (dh_next).
            dh = self.Wy.T @ dy + dh_next

            # Переводим градиент через tanh: производная tanh(z) = 1 - tanh(z)^2 = 1 - hs[t]^2. dh_raw — градиент по сумме до применения tanh.
            dh_raw = (1 - hs[t] ** 2) * dh

            # Градиент по Wx: dh_raw умноженный на входной символ xs[t], который был подан на этом шаге.
            dWx += dh_raw @ xs[t].T

            # Градиент по Wh: dh_raw умноженный на память с предыдущего шага (hs[t-1]), т.к. именно она умножалась на Wh.
            dWh += dh_raw @ hs[t - 1].T

            # Градиент по bh — совпадает с dh_raw (производная = 1, т.к. bh прибавляется напрямую).
            dbh += dh_raw

            # Передаём градиент на шаг назад (t-1): протаскиваем dh_raw через Wh^T. На следующей итерации это станет dh_next.
            dh_next = self.Wh.T @ dh_raw

        return dWx, dWh, dWy, dbh, dby

    def train_step(self, inputs, targets):
        """
        Один полный шаг обучения на одном куске текста (100 символов).
        Проходит все 4 этапа: forward -> backward -> clip -> обновление весов.
        """
        #    FORWARD: прогоняем кусок через модель, получаем loss (насколько
        #    сильно ошиблись) и промежуточные значения xs, hs, ps — они нужны
        #    backward, чтобы посчитать градиенты (без них backward не сможет
        #    восстановить, что происходило на каждом шаге t).
        loss, xs, hs, ps = self.forward(inputs, targets)

        #    BACKWARD: считаем градиенты — в какую сторону и насколько сильно
        #    нужно подвинуть каждую матрицу весов, чтобы в следующий раз
        #    ошибка (loss) была меньше.
        dWx, dWh, dWy, dbh, dby = self.backward(inputs, targets, xs, hs, ps)

        #    CLIP: обрезаем каждый градиент в диапазон [-5, 5]. Если градиент
        #    получился аномально большим (например, 850 вместо 2.3) — это
        #    "взрыв градиента" (exploding gradient). Без обрезки такой градиент,
        #    умноженный на LEARNING_RATE, сдвинул бы веса слишком резко за
        #    один шаг — модель могла бы "выстрелить" в область весов, откуда
        #    обучение уже не восстановится (loss улетит в NaN/inf).
        for dparam in [dWx, dWh, dWy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Обновление весов
        self.Wx -= LEARNING_RATE * dWx
        self.Wh -= LEARNING_RATE * dWh
        self.Wy -= LEARNING_RATE * dWy
        self.bh -= LEARNING_RATE * dbh
        self.by -= LEARNING_RATE * dby

        # Возвращаем loss, поделённый на длину куска — это средняя ошибка на один символ
        return loss / len(inputs)

    def _step(self, x_index, h):
        """
        Один шаг вперёд без сохранения истории (в отличие от forward).
        Принимает: номер текущего символа (x_index) и текущую память (h).
        Возвращает: (новая память, вероятности следующего символа).
        """
        x_onehot = np.zeros((self.vocab_size, 1))
        x_onehot[x_index] = 1

        h_new = np.tanh(self.Wx @ x_onehot + self.Wh @ h + self.bh)

        y_raw = self.Wy @ h_new + self.by
        exp_y = np.exp(y_raw - np.max(y_raw))
        probs = exp_y / np.sum(exp_y)

        return h_new, probs

    def generate(self, ch2i, i2ch, start_char, length=300):
        """
        Генерация текста, начиная с одного символа, через _step.
        """
        h = np.zeros((self.hidden_size, 1))
        x = ch2i[start_char]
        result = start_char

        for _ in range(length):
            h, probs = self._step(x, h)
            probs_flat = probs.ravel()  # превращаем (vocab_size, 1) в плоский массив для np.random.choice
            x = np.random.choice(range(self.vocab_size), p=probs_flat)

            result += i2ch[x]

        return result


    def prime(self, ch2i, prefix):
        """
        Прогоняет prefix через модель без обучения, по одному символу,
        чтобы получить состояние памяти h, "настроенное" на этот префикс.

        Возвращает: (h, x) — итоговая память и номер ПОСЛЕДНЕГО символа
        префикса (он же станет "текущим x" для начала генерации дальше).
        """
        h = np.zeros((self.hidden_size, 1))
        x = None

        for ch in prefix:
            x = ch2i[ch]
            h, probs = self._step(x, h)

        return h, x

    def complete_word(self, ch2i, i2ch, prefix, n_chars, mode='greedy'):
        """
        Достраивает prefix на n_chars новых символов.

        mode='greedy' -> всегда самый вероятный символ (np.argmax)
        mode='sample' -> случайно с учётом вероятностей (как в generate)
        """
        h, x = self.prime(ch2i, prefix)
        result = prefix

        for _ in range(n_chars):
            h, probs = self._step(x, h)
            probs_flat = probs.ravel()

            if mode == 'greedy':
                x = np.argmax(probs_flat)
            elif mode == 'sample':
                x = np.random.choice(range(self.vocab_size), p=probs_flat)

            result += i2ch[x]

        return result

    def save_weights(self):
        """
        Сохраняет все веса модели в папку FOLDER_WEIGHTS как .npy файлы.
        """
        np.save(f"{FOLDER_WEIGHTS}/Wx.npy", self.Wx)
        np.save(f"{FOLDER_WEIGHTS}/Wh.npy", self.Wh)
        np.save(f"{FOLDER_WEIGHTS}/Wy.npy", self.Wy)
        np.save(f"{FOLDER_WEIGHTS}/bh.npy", self.bh)
        np.save(f"{FOLDER_WEIGHTS}/by.npy", self.by)


    def load_weights(self):
        """
        Загружает все веса модели из папки FOLDER_WEIGHTS.
        """
        self.Wx = np.load(f"{FOLDER_WEIGHTS}/Wx.npy")
        self.Wh = np.load(f"{FOLDER_WEIGHTS}/Wh.npy")
        self.Wy = np.load(f"{FOLDER_WEIGHTS}/Wy.npy")
        self.bh = np.load(f"{FOLDER_WEIGHTS}/bh.npy")
        self.by = np.load(f"{FOLDER_WEIGHTS}/by.npy")

    def get_weights_copy(self):
        """Возвращает копию всех весов модели прямо сейчас."""
        return (self.Wx.copy(), self.Wh.copy(), self.Wy.copy(),
                self.bh.copy(), self.by.copy())

    def set_weights(self, weights):
        """Устанавливает веса модели из кортежа (Wx, Wh, Wy, bh, by)."""
        self.Wx, self.Wh, self.Wy, self.bh, self.by = weights