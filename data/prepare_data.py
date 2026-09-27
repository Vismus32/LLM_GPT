import glob
import dill

from Embeddings_process.BPE import BPE


def main():

    # Собираем все тексты
    all_text = []

    for file_path in glob.glob('text/pushkin_poetry/*.*'):
        file = open(file_path, 'r', encoding='utf8')
        text = file.read()
        all_text.append(text)

    all_text = '\n\n\n'.join(all_text)

    print('Количество символов:', len(all_text))

    # Обучаем BPE
    bpe = BPE(2000)
    bpe.fit(all_text)

    # Сохраняем обученный токенизатор
    bpe.save('data/bpe.dill')

    # Кодируем весь корпус в токены
    tokens = bpe.encode(all_text)

    print('Количество токенов:', len(tokens))

    # Сохраняем токены
    with open('data/tokens.dill', 'wb') as file:
        dill.dump(tokens, file)


if __name__ == '__main__':
    main()