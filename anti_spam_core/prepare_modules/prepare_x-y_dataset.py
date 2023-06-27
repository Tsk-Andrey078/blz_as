import pandas as pd
import pymorphy2
import numpy as np
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class PrepareDataXY:
    def __init__(self) -> None:
        pass
    
    def delete_null(self, dataframe):
        dataframe.replace('', np.nan, inplace=True)
        df_filtered = dataframe.dropna()
        return df_filtered

    def preprocessing(self, text):
        text = re.sub('<[^>]*>', '', text)
        emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) +  " ".join(emotions).replace('-', '')
        return text

    def stemmer(self, text):
        morph_ru = pymorphy2.MorphAnalyzer(lang='ru')
        # Обработка слов на русском языке
        parsed_word = morph_ru.parse(text)[0]
        lemma = parsed_word.normal_form
        return lemma
            
    def tokenizer_porter(self, text):
        return list(map(self.stemmer, text.split()))

    def get_dataframe(self, name):
        df = pd.DataFrame()
        df = pd.read_csv(name)
        return df

    def process_text(self, text):
        return self.tokenizer_porter(self.preprocessing(text))

if __name__ == "__main__":
    pr = PrepareDataXY()
    df = pr.get_dataframe("./posts_inst_preready.csv")

    df_f = pr.delete_null(df)
    texts = df_f['text'].tolist()
    
    # Инициализация пула процессов
    num_processes = cpu_count()  # Количество доступных процессоров
    with Pool(num_processes) as pool:
        # Применение функции process_text к каждому тексту с использованием мультипроцессинга
        processed_texts = list(tqdm(pool.imap(pr.process_text, texts), total=len(texts)))
    
    # Обновление столбца 'text' с обработанными текстами
    df_f['text'] = processed_texts
    
    df_f.drop_duplicates(subset='text')
    df_f.to_csv('./posts_inst_ready.csv', index=False)