from rnn.config import DATASET_PATH, EPOCHS
from rnn.data import load_and_clean, build_vocab, make_sequences, shuffle_pairs
from rnn.model import RNN


clean_data = load_and_clean(DATASET_PATH)

vocab_size, chars, ch2i, i2ch = build_vocab(clean_data)

all_inputs, all_targets = make_sequences(clean_data, ch2i)

all_inputs, all_targets = shuffle_pairs(all_inputs, all_targets)

model = RNN(vocab_size)

best_loss = float('inf')
best_weights = None

for epoch in range(EPOCHS):
    total_loss = 0

    for i in range(len(all_inputs)):
        loss = model.train_step(all_inputs[i], all_targets[i])
        total_loss += loss

        if i % 500 == 0:
            print(f"Эпоха {epoch + 1}, шаг {i}, потеря: {loss:.2f}")

        avg_loss = total_loss / (i + 1)

        # Сохраняем если вес лучше предыдущего
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = (model.get_weights_copy())

    print(f"Эпоха {epoch + 1} завершена, ср. потеря: {total_loss / len(all_inputs):.3f}")

model.set_weights(best_weights)
model.save_weights()
print("Обучение закончено")
print("Средняя потеря:", best_loss)