import pandas as pd
import pymorphy2
import numpy as np
import re
from tqdm import tqdm

class PrepareDataX:
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

""" if __name__ == "__main__":
    pr = PrepareDataX()
    df = pr.get_dataframe("./posts_normal_pd.csv")

    df_f = pr.delete_null(df)
    df_f['text'] = df_f['text'].apply(pr.preprocessing)
    
    
    df_f.drop_duplicates(subset='text')
    df_f.to_csv('./posts_normal_pd_r.csv', index=False) """