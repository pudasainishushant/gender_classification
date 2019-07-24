import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Predict:
    def __init__(self):
        with open('./serializer/male_female_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        with open('./serializer/male_female_identifier.pickle', 'rb') as handle:
            self.loaded_model = pickle.load(handle)

    def predict_single(self,name):
        text = np.array([name])
        sequences_test = self.tokenizer.texts_to_sequences(text)
        sequence = pad_sequences(sequences_test,maxlen = 10)
        prediction = self.loaded_model.predict(sequence)
        if (prediction>0.45):
            return("male")
        else:
            return("female")

obj2 = Predict()
text_input = str(input("Enter the name to be classified :: "))
output_gender = obj2.predict_single(text_input)
print(output_gender)