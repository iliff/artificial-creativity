"""
A very simple beginning tutorial to machine translation with neural networks.
Need to test on a larger data set to be sure I have not overlooked anything.
Start with sentence represented by a list of numbers, where every number
represents a word. Outputs a sequence of numbers which would represent a
target language sentence. We can refine this to make it more understandable.
And we can demonstrate how several viable sentences can come from a single
source language starting point.

Creator: Justin (jbarber@iliff.edu) :-)
"""

import numpy as np
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.utils import to_categorical

np.random.seed(42)

#######################################################################
# DATA: 4 SENTENCES PER LANGUAGE AND NEXT-STEP SENTENCES FOR ENGLISH
#######################################################################

# 4 HEBREW SENTENCES; 7 IS MAX HEBREW NUMBER
# ברא את השמים
# ברא את הארץ
# את השמים
# את הארץ
hebrew_texts = np.array([3, 4, 5, 0, 0] +
                        [3, 4, 6, 0, 0] +
                        [4, 5, 0, 0, 0] +
                        [4, 6, 0, 0, 0]).reshape((4, 5))

# 4 ENGLISH SENTENCES; 8 IS MAX HEBREW NUMBER
english_texts = np.array([1, 3, 4, 5, 6, 2] +
                         [1, 3, 4, 5, 7, 2] +
                         [1, 5, 6, 2, 0, 0] +
                         [1, 5, 7, 2, 0, 0]).reshape((4, 6))

# 4 ENGLISH SENTENCES ONE STEP AHEAD OF ENGLISH SENTENCES ABOVE
next_step_english_texts = np.array([3, 4, 5, 6, 2, 0] +
                                   [3, 4, 5, 7, 2, 0] +
                                   [5, 6, 2, 0, 0, 0] +
                                   [5, 7, 2, 0, 0, 0]).reshape((4, 6))

#######################################################################
# MODEL 1 (FOR TRAINING ONLY): GIVEN A HEBREW SENTENCE AND AN ENGLISH
#   SENTENCE, PREDICT NEXT-STEP ENGLISH SENTENCE
#######################################################################

# HEBREW INPUTS AND TRAINING; THE HEBREW PORTION WILL ALSO CONSTITUTE MODEL 2 BELOW
hebrew_inputs = Input(shape=(None,), name='hebrew_inputs')
hebrew_embedding = Embedding(input_dim=7, output_dim=5, name='hebrew_embedding')(hebrew_inputs)
_, hebrew_hidden_state, hebrew_cell_state = LSTM(256, return_sequences=True, return_state=True, name='hebrew_lstm')(hebrew_embedding)
hebrew_states = [hebrew_hidden_state, hebrew_cell_state]

# TARGET LANGUAGE INPUTS AND TRAINING
english_inputs = Input(shape=(None,), name='english_inputs')
english_embedding = Embedding(input_dim=8, output_dim=5, name='english_embedding')(english_inputs)
sequenced_english_lstm = LSTM(256, return_sequences=True,
                              name='sequence_english_lstm')(english_embedding, initial_state=hebrew_states)
english_outputs = Dense(8, activation='softmax')(sequenced_english_lstm)

model = Model(inputs=[hebrew_inputs, english_inputs], outputs=english_outputs)

print(model.summary(line_length=120))

#######################################################################
# COMPILE AND FIT MODEL 1
#######################################################################

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ONE-HOT ENCODE NEXT-STEP ARRAYS
categorical_next_step = to_categorical(next_step_english_texts).reshape((4, 6, 8))
model.fit(x=[hebrew_texts, english_texts], y=categorical_next_step, epochs=500,
          batch_size=None)

#######################################################################
# TEST MODEL 1 WITH PREDICTION
#######################################################################

# TEST BY TRYING TO PREDICT NEXT STEP ARRAY BASED UPON 1ST ROW OF HEBREW AND 1ST ROW OF ENGLISH
predictions = model.predict(x=[hebrew_texts[1].reshape((1, 5)), english_texts[1].reshape((1, 6))])
print([np.argmax(a) for a in predictions[0]])  # should be close to [3, 4, 5, 7, 2, 0]

#######################################################################
# MODEL 2 (INFERENCE 1): GIVEN HEBREW INPUTS, PREDICT A CORRESPONDING
#   HIDDEN STATE AND CELL STATE
#######################################################################

# HEBREW MODEL
hebrew_model = Model(inputs=hebrew_inputs, outputs=hebrew_states)

# TEST HEBREW MODEL WITH A PREDICTION
inference_prediction_hebrew = hebrew_model.predict(x=hebrew_texts[2].reshape((1, 5)))
print(inference_prediction_hebrew)  # prints arrays: [array.shape == (1, 256), array.shape == (1, 256)]

#######################################################################
# MODEL 3 (INFERENCE 2): GIVEN AN ENGLISH STARTING POINT (INPUT),
#   HEBREW/ENGLISH HIDDEN STATE, AND HEBREW/ENGLISH CELL STATE, PREDICT
#   AN ENGLISH NEXT-STEP OUTPUT, A NEW HEBREW/ENGLISH HIDDEN STATE, AND
#   A NEW HEBREW/ENGLISH CELL STATE
#######################################################################

hidden_english_input = Input(shape=(256,))
cell_english_input = Input(shape=(256,))
english_input_states = [hidden_english_input, cell_english_input]

sequenced_english_lstm, hidden_english_lstm, cell_english_lstm = LSTM(256, return_state=True,
                                                                      name='sequence_english_lstm')(english_embedding,
                                                                                                    initial_state=english_input_states)
english_output_states = [hidden_english_lstm, cell_english_lstm]
english_outputs = Dense(8, activation='softmax')(sequenced_english_lstm)

# `ENGLISH_INPUTS` AND `ENGLISH_OUTPUTS` == ORIGINAL INPUT AND OUTPUT OF MODEL 1
english_model = Model(inputs=[english_inputs] + english_input_states,
                      outputs=[english_outputs] + english_output_states)

print(english_model.summary(line_length=120))

#######################################################################
# MAKE A PREDICTION IN TWO STEPS: FIRST, PREDICT THE HEBREW HIDDEN AND
#   CELL STATES. THEN, WITH THAT OUTPUT AS INPUT TO THE NEXT MODEL,
#   PREDICT THE NEXT-STEP ENGLISH WITHOUT REQUIRING A FULL ENGLISH
#   TRANSLATION AS INPUT. (ONLY AN ENGLISH START IS REQUIRED.)
#######################################################################

# NOW PREDICT `HEBREW_STATES` [HEBREW_HIDDEN_STATE, HEBREW_CELL_STATE]
heb_hidden_prediction, heb_cell_prediction = hebrew_model.predict(hebrew_texts[[1]].reshape((1, 5)))
print('hebrew hidden prediction =>', heb_hidden_prediction)
print('hebrew cell prediction =>', heb_cell_prediction)

# NOW USE THOSE STATE PREDICTIONS AS THE INPUTS TO THE ENGLISH MODEL
state = [1, 5]
for i in range(2, 6):
    english_prediction, eng_hidden_state, eng_cell_state = english_model.predict(
        x=[np.array(state).reshape(1, i), heb_hidden_prediction, heb_cell_prediction])

    # [4, 5, 0, 0, 0] => [1, 5, 6, 2, 0, 0]
    print('english prediction =>', english_prediction)

    state.append(np.argmax(english_prediction[0]))
    heb_hidden_prediction, heb_cell_prediction = eng_hidden_state, eng_cell_state

print(state)
