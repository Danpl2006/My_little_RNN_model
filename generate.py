from rnn.config import DATASET_PATH
from rnn.data import build_vocab, load_and_clean
from rnn.model import RNN


text = load_and_clean(DATASET_PATH)


vocab_size, chars, ch2i, i2ch = build_vocab(text)

model = RNN(vocab_size)
model.load_weights()

result = model.generate(ch2i, i2ch, start_char='а', length=300)
print(result)