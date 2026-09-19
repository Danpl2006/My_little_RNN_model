from rnn.config import DATASET_PATH
from rnn.data import load_and_clean, build_vocab
from rnn.model import RNN


text = load_and_clean(DATASET_PATH)
vocab_size, chars, ch2i, i2ch = build_vocab(text)
model = RNN(vocab_size)
model.load_weights()

mode_input = input("Выбор режима (1 - жадный, 2 - случайный): ")
if mode_input == '1':
   mode = 'greedy'
else:
   mode = 'sample'

prefix = input("Введите слово: ")
while prefix.strip() == "":
    print("Пустое поле!")
    prefix = input("Введите слово: ")

while True:
    n_chars_raw = input("Введите количество символов для продолжения: ")
    try:
        n_chars = int(n_chars_raw)
    except ValueError:
        print("Нужно ввести целое число")
        continue   # Начать цикл заново, не проверяя дальше

    if n_chars <= 0:
        print("Количество символов должно быть положительным")
        continue
    break

try:
    result = model.complete_word(ch2i, i2ch, prefix, n_chars, mode)
    print("Результат:", result)
except KeyError:
    print("В слове есть символ, которого нет в словаре модели(только русские буквы, знаки препинания и пробел)")