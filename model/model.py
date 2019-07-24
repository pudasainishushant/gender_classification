import pandas as pd
import numpy as np
import pickle


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,LSTM, Dropout, Activation
from keras.layers import BatchNormalization






vocabulary_size = 20000

class ModelName:
	def __init__(self,male_name_path,female_name_path):
		self.male_name_df = pd.read_csv(male_name_path,sep="\t")
		self.female_name_df= pd.read_csv(female_name_path,sep="\t")
		

	def train_model(self):
		final_df = self.male_name_df.append(self.female_name_df, ignore_index=True)
		final_df=final_df.sample(frac=1).reset_index(drop=True)
		final_df.loc[:,"Name"] = final_df.Name.apply(lambda x : str(x))
		final_df["Label"] = final_df["Label"].replace(["male" , "female"] , [1 , 0])
		labels = final_df["Label"]

		#Building tokenizer
		tokenizer = Tokenizer(char_level=True, oov_token='UNK',num_words = vocabulary_size)
		tokenizer.fit_on_texts(final_df["Name"])
		sequences = tokenizer.texts_to_sequences(final_df['Name'])
		data = pad_sequences(sequences,maxlen = 10)

		#Build model
		model = Sequential()
		model.add(Embedding(20000, 50, input_length=10))
		model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))
		model.add(Dense(50, activation="tanh"))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))
		model.add(Dense(50, activation="tanh"))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(data, np.array(labels), validation_split=0.4, epochs=5)
		
		
		# saving trained model
		with open('../serializer/male_female_identifier.pickle', 'wb') as handle:
			pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# saving tokenizer
		with open('../serializer/male_female_tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


obj1 = ModelName("/home/shushant/Desktop/name_gender_classification/profanity/profanity-detection/data/male.csv","/home/shushant/Desktop/name_gender_classification/profanity/profanity-detection/data/female.csv")
obj1.train_model()