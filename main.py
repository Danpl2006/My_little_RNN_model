import numpy as np
import re

# Открытие датасета
text = open('datasets/master_dataset.txt', encoding='cp1251').read()

# Нужные символы
allowed = 'ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁйцукенгшщзхъфывапролджэячсмитьбюё,.!?:;-— \n'

# Очищенный датасет
clean = ''.join(c for c in text if c in allowed)
# Чистка пробелов и переносов
clean = re.sub(r' +', ' ', clean)
clean = re.sub(r'\n+', '\n', clean)
clean = clean.strip()

# print(len(clean))
# print(clean[:200])

chars = sorted(set(clean))
vocab_size = len(chars)
print(vocab_size)
print(chars)