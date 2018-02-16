import re

import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dense, Embedding, Input, LSTM, regularizers
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

np.random.seed(42)

#######################################################################
# TRAINING DATA: HEBREW SENTENCES, ENGLISH SENTENCES, AND
# NEXT-STEP ENGLISH SENTENCES
#######################################################################

df = pd.read_csv('data/EngHeb/tabbed_heb_eng_corpus2.txt', sep='\t', names=['hebrew', 'english'])

# remove hebrew letters ס and פ; for some reason the last row lacks the english translation,
# so remove that row as well
df['hebrew'] = df['hebrew'].apply(lambda s: re.sub(r'(^|\s+)[פס](?=\s+|$)', '', s))
df = df.iloc[:-1]

# add english sov (start of verse) and eov (end of verse)
df['english'] = df['english'].apply(lambda s: 'sov ' + s + ' eov')

# reduce df for machine-without-gpu testing
df = df.sample(frac=0.01)

hebrew_tokenizer = Tokenizer()
hebrew_tokenizer.fit_on_texts(df.hebrew)
heb_seqs = hebrew_tokenizer.texts_to_sequences(df.hebrew)
heb_pad_seqs = pad_sequences(heb_seqs)
heb_sent_len = heb_pad_seqs.shape[1]

english_tokenizer = Tokenizer()
english_tokenizer.fit_on_texts(df.english)
eng_seqs = english_tokenizer.texts_to_sequences(df.english)
eng_pad_seqs = pad_sequences(eng_seqs)
eng_sent_len = eng_pad_seqs.shape[1]

input_length = max(heb_sent_len, eng_sent_len)

next_step_eng_seqs = [seq[1:] + [0] for seq in eng_seqs]
next_step_eng_pad_seqs = pad_sequences(next_step_eng_seqs)
next_step_eng_cat = to_categorical(next_step_eng_pad_seqs)

cat_reshaped = next_step_eng_cat.reshape((eng_pad_seqs.shape[0],
                                          eng_pad_seqs.shape[1],
                                          next_step_eng_cat.shape[1]))

#######################################################################
# MODEL 1: GIVEN A HEBREW SENTENCE AND AN ENGLISH
#   SENTENCE, PREDICT NEXT-STEP ENGLISH SENTENCE
#######################################################################

# HEBREW INPUTS AND TRAINING
hebrew_inputs = Input(shape=(None,), name='hebrew_inputs')
hebrew_embedding = Embedding(input_dim=len(hebrew_tokenizer.word_index) + 1,
                             output_dim=512, name='hebrew_embedding')(hebrew_inputs)
heb_lstm_1 = LSTM(1024, activation='relu', return_sequences=True,
                  kernel_regularizer=regularizers.l2(0.001))(hebrew_embedding)
heb_lstm_2 = LSTM(1024, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(heb_lstm_1)
heb_lstm_3 = LSTM(1024, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(heb_lstm_2)
_, hebrew_hidden_state, hebrew_cell_state = LSTM(1024, return_sequences=True, return_state=True, name='hebrew_lstm',
                                                 kernel_regularizer=regularizers.l2(0.001))(heb_lstm_3)
hebrew_states = [hebrew_hidden_state, hebrew_cell_state]

# ENGLISH INPUTS AND TRAINING
english_inputs = Input(shape=(None,), name='english_inputs')
english_embedding = Embedding(input_dim=len(english_tokenizer.word_index) + 1,
                              output_dim=512, name='english_embedding')(english_inputs)
eng_lstm_1 = LSTM(1024, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(
    english_embedding)
eng_lstm_2 = LSTM(1024, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(eng_lstm_1)
eng_lstm_3 = LSTM(1024, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(eng_lstm_2)
sequenced_english_lstm = LSTM(1024, return_sequences=True, kernel_regularizer=regularizers.l2(0.001),
                              name='sequence_english_lstm')(english_embedding, initial_state=hebrew_states)
english_outputs = Dense(len(english_tokenizer.word_index) + 1, activation='softmax')(eng_lstm_3)

model = Model(inputs=[hebrew_inputs, english_inputs], outputs=english_outputs)

print(model.summary(line_length=120))

#######################################################################
# COMPILE AND FIT MODEL 1
#######################################################################

adam = optimizers.Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# ONE-HOT ENCODE NEXT-STEP ARRAYS
model.fit(x=[heb_pad_seqs, eng_pad_seqs], y=cat_reshaped, epochs=12,
          batch_size=16)

model.save('layeredmodel.h5')

#######################################################################
# MODEL 2 (INFERENCE 1): GIVEN HEBREW INPUTS, PREDICT A CORRESPONDING
#   TRANSLATION HIDDEN STATE AND CELL STATE
#######################################################################

# HEBREW MODEL
hebrew_model = Model(inputs=hebrew_inputs, outputs=hebrew_states)

#######################################################################
# MODEL 3 (INFERENCE 2): GIVEN AN ENGLISH STARTING POINT (INPUT),
#   TRANSLATION HIDDEN STATE, AND TRANSLATION CELL STATE, PREDICT AN
#   ENGLISH NEXT-STEP OUTPUT, A NEW TRANSLATION HIDDEN STATE, AND A NEW
#   TRANSLATION CELL STATE
#######################################################################

hidden_english_input = Input(shape=(1024,))
cell_english_input = Input(shape=(1024,))
english_input_states = [hidden_english_input, cell_english_input]

# ``eng_lstm_3`` was originally ``english_embedding`` in the 2nd line after this one!!!
sequenced_english_lstm, hidden_english_lstm, cell_english_lstm = LSTM(1024, return_state=True,
                                                                      name='sequence_english_lstm')(eng_lstm_3,
                                                                                                    initial_state=english_input_states)
english_output_states = [hidden_english_lstm, cell_english_lstm]
english_outputs = Dense(len(english_tokenizer.word_index) + 1, activation='softmax')(sequenced_english_lstm)

# `ENGLISH_INPUTS` AND `ENGLISH_OUTPUTS` == ORIGINAL INPUT AND OUTPUT OF MODEL 1
english_model = Model(inputs=[english_inputs] + english_input_states,
                      outputs=[english_outputs] + english_output_states)

print(english_model.summary(line_length=120))

#######################################################################
# MAKE A PREDICTION IN TWO STEPS: FIRST, PREDICT THE HEBREW HIDDEN AND
#   CELL STATES. THEN, WITH THAT OUTPUT AS INPUT TO THE NEXT MODEL,
#   PREDICT THE NEXT-STEP ENGLISH WITHOUT REQUIRING A FULL ENGLISH
#   TRANSLATION AS INPUT. (ONLY AN ENGLISH START IS REQUIRED.)
#   THIS WILL BECOME A FUNCTION EVENTUALLY.
#######################################################################

# GET A HEBREW SAMPLE FROM PADDED SEQUENCES FOR TESTING
hebrew_sample = heb_pad_seqs[[1]].reshape((1, -1))

# PREDICT `HEBREW_STATES` [HEBREW_HIDDEN_STATE, HEBREW_CELL_STATE]
heb_hidden_prediction, heb_cell_prediction = hebrew_model.predict(hebrew_sample)

# NOW USE THOSE STATE PREDICTIONS AS THE INPUTS TO THE ENGLISH MODEL
heb_hidden_prediction, heb_cell_prediction = hebrew_model.predict(hebrew_sample)
state = [english_tokenizer.word_index['sov']]

while len(state) < eng_pad_seqs.shape[1]:
    english_prediction, eng_hidden_state, eng_cell_state = english_model.predict(
        x=[np.array(state).reshape((1, -1)), heb_hidden_prediction, heb_cell_prediction])

    state.append(np.argmax(english_prediction[0]))
    heb_hidden_prediction, heb_cell_prediction = eng_hidden_state, eng_cell_state

    if state[-1] == english_tokenizer.word_index['eov']:
        break

indexed_words = {v: k for k, v in english_tokenizer.word_index.items()}
print(df.iloc[1].hebrew, '=>', df.iloc[1].english)
print([indexed_words.get(s) for s in state])
