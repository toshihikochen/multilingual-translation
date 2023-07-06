import os
import warnings

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

from utils import auglib as A
from utils.bidict import bidict

# disable the advisory warning of transformers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

punctuations = [
    '.', ',', '!', '?', '。', '，', '！', '？', '、', '・', '「', '」', '『', '』', '（', '）', '(', ')', '[', ']',
    '【', '】', '《', '》', '〈', '〉', '…', '—', '——', '～', '〜', '\'', '"', '‘', '’', '“', '”', '‥', '〝', '〟',
    '〰', '〾', '〿', '–', '‒', '−', '〜', '～', '々', '‖'
]


# swap UPPER and lower case in english
def switch_case(text):
    new_text = ''
    for i, c in enumerate(text):
        if c.isupper():
            new_text += c.lower()
        elif c.islower():
            new_text += c.upper()
        else:
            new_text += c
    return new_text


# swap katakana and hiragana in japanese
bidict_ja = bidict({
    'あ': 'ア', 'い': 'イ', 'う': 'ウ', 'え': 'エ', 'お': 'オ',
    'か': 'カ', 'き': 'キ', 'く': 'ク', 'け': 'ケ', 'こ': 'コ',
    'さ': 'サ', 'し': 'シ', 'す': 'ス', 'せ': 'セ', 'そ': 'ソ',
    'た': 'タ', 'ち': 'チ', 'つ': 'ツ', 'て': 'テ', 'と': 'ト',
    'な': 'ナ', 'に': 'ニ', 'ぬ': 'ウ', 'ね': 'ネ', 'の': 'ノ',
    'は': 'ハ', 'ひ': 'ヒ', 'ふ': 'フ', 'へ': 'ヘ', 'ほ': 'ホ',
    'ら': 'ラ', 'り': 'リ', 'る': 'ル', 'れ': 'レ', 'ろ': 'ロ',
    'わ': 'ワ', 'を': 'ヲ', 'だ': 'ダ', 'ぢ': 'ヂ', 'づ': 'ヅ',
    'で': 'デ', 'ど': 'ド', 'ぁ': 'ァ', 'ぃ': 'ィ', 'ぅ': 'ゥ',
    'ぇ': 'ェ', 'ぉ': 'ォ', 'ん': 'ン', 'っ': 'ッ',
})


def switch_kana(text):
    new_text = ''
    for i, c in enumerate(text):
        if c in bidict_ja:
            new_text += bidict_ja[c]
        else:
            new_text += c
    return new_text


# augment methods on specific language
english_augment = A.Compose([
    A.ReplaceWord(num_replace=1, replace_func=switch_case, p=0.2),
    A.NeighborSwapWord(num_swap=1, p=0.1),
    A.InsertWordFromOther(num_insert=1, other=punctuations, p=0.1),
    A.InsertWordFromSelf(num_insert=1, p=0.01),
])
japanese_augment = A.Compose([
    A.ReplaceWord(num_replace=1, replace_func=switch_kana, p=0.2),
    A.NeighborSwapWord(num_swap=1, p=0.1),
    A.InsertWordFromOther(num_insert=1, other=punctuations, p=0.1),
    A.InsertWordFromSelf(num_insert=1, p=0.01),
])
chinese_augment = A.Compose([
    A.NeighborSwapWord(num_swap=1, p=0.1),
    A.InsertWordFromOther(num_insert=1, other=punctuations, p=0.1),
    A.InsertWordFromSelf(num_insert=1, p=0.01),
])


# augment function
def augment(text, lang):
    if lang == 'en':
        return english_augment(text)
    elif lang == 'ja':
        return japanese_augment(text)
    elif lang == 'zh':
        return chinese_augment(text)
    else:
        return text


class ParquetMatrix(Dataset):
    def __init__(self, parquet_path, tokenizer=None, use_augment=False, specific_lang_pair=None):
        # judge if the path is folder or file
        if os.path.isdir(parquet_path):
            self.data = pd.DataFrame()
            for file in os.listdir(parquet_path):
                if file.endswith('.parquet'):
                    df = pd.read_parquet(os.path.join(parquet_path, file), engine='pyarrow')
                    self.data = pd.concat([self.data, df], ignore_index=True)
            if len(self.data) == 0:
                warnings.warn(f'No parquet file found in {parquet_path}')
        elif os.path.isfile(parquet_path):
            self.data = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f'No such file or directory: {parquet_path}')

        self.tokenizer = tokenizer
        self.use_augment = use_augment
        self.specific_lang_pair = specific_lang_pair

        self.langs = self.data.columns.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get the source and target text
        row = self.data.iloc[idx]

        # select source and target language
        if self.specific_lang_pair is not None:
            source_lang, target_lang = self.specific_lang_pair
        else:
            # remove columns that corresponding value is None
            columns = [col for col in self.langs if row[col] is not None]
            # randomly select source and target language
            source_lang, target_lang = np.random.choice(columns, size=2, replace=False)
        source_text = row[source_lang]
        target_text = row[target_lang]

        # add translation tag to the source text
        translation_tag = f'<{source_lang}2{target_lang}>'

        # tokenize the source and target texts if tokenizer is provided
        # otherwise, return the source and target texts
        if self.tokenizer is None:
            return source_text, target_text

        if self.use_augment:
            # use tokenizer to tokenize the source and target texts
            source_tokens = self.tokenizer.tokenize(translation_tag + source_text)
            # augment the source text, don't augment the translation tag
            source_tokens = [source_tokens[0]] + augment(source_tokens[1:], source_lang)
            # convert the tokens to ids
            source_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(source_tokens))
            # create attention mask
            source_mask = torch.ones(len(source_ids), dtype=torch.long)

            # tokenize the target text
            target_tokens = self.tokenizer.tokenize(target_text)
            # augment the target text
            target_tokens = augment(target_tokens, target_lang)
            # convert the tokens to ids
            target_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(target_tokens))

        else:
            tokens = self.tokenizer(translation_tag + source_text, text_targets=target_text)
            source_ids = tokens['input_ids'].squeeze()
            source_mask = tokens['attention_mask'].squeeze()
            target_ids = tokens['labels'].squeeze()

        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': target_ids
        }

    def get_langs(self):
        return self.langs

    def get_lengths(self):
        return self.data.count()
