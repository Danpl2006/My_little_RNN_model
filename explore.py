from collections import Counter
import matplotlib.pyplot as plt
from rnn.config import DATASET_PATH
from rnn.data import load_and_clean, build_vocab


def get_letter_frequency(text, top_n=20):
    """
    Считает, сколько раз встречается каждый символ в тексте,
    и возвращает top_n самых частых как список пар (символ, количество).
    """
    counts = Counter(text)
    top_letters = counts.most_common(top_n)
    return top_letters


def print_basic_stats(text, vocab_size, chars):
    """
    Печатает базовую информацию о датасете: размер, словарь.
    """
    print("Всего символов: ", len(text))
    print( "Размер словаря: ", vocab_size)
    print("Символы словаря: ", chars)


def plot_letter_frequency(top_letters):
    """
    Рисует столбчатую диаграмму частоты букв и сохраняет как .png.
    """
    # распаковать top_letters на два отдельных списка: letters, counts
    letters = [i[0] for i in  top_letters]
    counts = [i[1] for i in  top_letters]

    plt.bar(letters, counts)
    plt.title("Частота символов в датасете")
    plt.xlabel("Символ")
    plt.ylabel("Количество")
    plt.savefig("letter_frequency.png")
    plt.show()


def main():
    text = load_and_clean(DATASET_PATH)

    vocab_size, chars, ch2i, i2ch = build_vocab(text)

    print_basic_stats(text, vocab_size, chars)

    top_letters = get_letter_frequency(text)
    plot_letter_frequency(top_letters)


if __name__ == '__main__':
    main()