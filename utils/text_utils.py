import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def tokenize_captions(captions_file, top_k=5000):
    df = pd.read_csv(captions_file)
    captions = df['caption'].apply(lambda x: 'startseq ' + x + ' endseq')

    tokenizer = Tokenizer(num_words=top_k, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)

    # Save tokenizer
    with open('../data/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max(len(x) for x in sequences)

    return tokenizer, max_length
