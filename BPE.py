#Byte-pair encoder
from typing import List
import dill
class BPE():
    """Токенизатор Byte-Pair Encoding (BPE).

    Реализует простой алгоритм BPE, который начинает с отдельных символов
    в качестве токенов и итеративно объединяет наиболее часто встречающуюся
    соседнюю пару токенов до тех пор, пока размер словаря не достигнет заданногозначения. 

    Атрибуты: 
        vocab_size (int): Целевой размер словаря.
        id2token (dict[int, str]): Отображение идентификаторов токенов на соответствующие токены.
        token2id (dict[str, int]): Отображение токенов на их идентификаторы. 
    """
    def __init__(self,vocab_size:int):
        self.vocab_size = vocab_size 
        self.id2token = {} #Словарь ID -> токен
        self.token2id = {} #Словарь токен -> ID

    def fit(self, text:str):
        """Обучает токенизатор BPE на переданном тексте.

        Процесс продолжается до тех пор, пока размер словаря
        не достигнет заданного значения `vocab_size`.

        Args:
            text (str): Текст, используемый для обучения
                токенизатора.

        Returns:
            None: Обученный словарь сохраняется в атрибутах
                `id2token` и `token2id`.
        """

        uniq_tokens = []
        tokens = []

        uniq_tokens = sorted(set(text))
        tokens = list(text)
        while len(uniq_tokens) != self.vocab_size:
            pairs = {}
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i + 1])
                if pair in pairs:
                    pairs[pair]+=1
                else:
                    pairs[pair] = 1
            max_pair = None
            max_count = 0
            for pair in pairs:
                if pairs[pair] > max_count:
                    max_count = pairs[pair]
                    max_pair = pair

            new_token = ''.join(max_pair)
            uniq_tokens.append(new_token)

            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == max_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        self.id2token = {i: token for i, token in enumerate(uniq_tokens)}

        self.token2id = {token: i for i, token in enumerate(uniq_tokens)}

    def encode(self, text:str) -> List[int]:
        """
        Энкодер

        Кодирует строку.
        Важно: Выбрал не самый оптимальный вариант, с последовательной заменой 
        т.к. нет мощностей чтобы сделать полноценный вариант с заменой сразу всех схожих символов

        Args:
            text (str): Текст, который нужно закодировать

        Returns:
            encode_text (list): Закодированный список исходного текста

        """
        simbols = list(text)
        i = 0
        vocab = self.token2id
        new_simbols = []
        while i < len(simbols):
            words = []
            for tokens in vocab:
                if simbols[i] == tokens[0]:
                    words.append(tokens)
            words.sort(key=len, reverse=True) #Сортируем по длине
            for tokens in words:
                if simbols[i:i+len(tokens)] == list(tokens):
                    i += len(tokens)
                    new_simbols.append(tokens)
                    break

        encode_text = []

        for token in new_simbols:
            encode_text.append(vocab[token])
            
        return encode_text
    def decode(self, token_ids : List[int]) -> str:
        """
        Декодер

        Заменяет полученные идентификаторы токенов на их текстовые представления

        """
        
        id2token = self.id2token
        tokens = []

        for token_id in token_ids:
            tokens.append(id2token[token_id])

        return ''.join(tokens)


    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

        print(f"Объект сохранён в {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = dill.load(f)

        print(f"Объект загружен из {filename}")
        return obj


# if __name__ == "__main__":
#     text = (
#         'Однажды был случай в далёком Макао: '
#         'макака коалу в какао макала, коала лениво какао лакала, '
#         'макака макала, коала икала.'
#     )

#     print("text:")
#     print(text)

#     BP = BPE(30)

#     BP.fit(text)
#     print('\nVocab:')
#     print(BP.id2token.items())

#     encoded = BP.encode(text)
#     print("\nEncode list:")
#     print(encoded)

#     decoded = BP.decode(encoded)
#     print("\nDecode text")
#     print(decoded)

#     if text == decoded:
#         print("Всё работает верно")

if __name__ == "__main__":
    text = (
        'Однажды был случай в далёком Макао: '
        'макака коалу в какао макала, коала лениво какао лакала, '
        'макака макала, коала икала.'
    )
#     BP = BPE(30)
#     BP.fit(text)

#     BP.save('data/bpe.dill')

    BP2 = BPE.load('data/bpe.dill')

    encoded = BP2.encode(text)
    print(encoded)
    decoded = BP2.decode(encoded)

    print(decoded)
    print(decoded == text)


