import speech_recognition as sr
import pyttsx3 as speak

#system control
import os
#import playsound
import numpy as np
import time
import sys
import ctypes
import wikipedia
import datetime
import json
import re
import webbrowser
import smtplib
import requests
import urllib
import urllib.request as urllib2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import strftime

##Google text to speak
#from gtts import gTTS

#from youtube_search import YoutubeSearch
from youtube_search import YoutubeSearch

#NLP libs
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import treebank


#https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281
import pickle
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# from underthesea import vn
#nltk.download('stopwords')

class Action():
	"""docstring for Action"""
	def __init__(self, keyword, script, notify):
		super(Action, self).__init__()
		self.keyword = keyword
		self.script = script
		self.notify = notify
		self.status = ""
		self.active = False
		
class Robot():
	"""docstring for Robot"""
	def __init__(self, nltk, wikipedia, record_language):
		super(Robot, self).__init__()
		self.name = "Zira"
		self.speaker = speak.init()
		self.recognizer = sr.Recognizer()
		self.nltk = nltk
		self.record_language = record_language
		self.voiceID = 2
		self.none = "Can you speak again!"
		self.actions = []
		self.dataStorage = "AIDataStorage.txt"
		self.news_fields = self.getFields("news")
		self.wiki_fields = self.getFields("wikipedia")

		#upgrade
		#self.wikipedia = wikipedia.set_lang('vi')
		self.wikipedia = wikipedia
		self.language_2 = 'vi'
		#self.path = ChromeDriverManager().install()

		#Data
		self.training_sentences = []
		self.training_labels = []
		self.labels = []
		self.responses = []
		self.data = None

		#Model
		self.model = None
		self.max_len = 20
		self.tokenizer = None
		self.lbl_encoder = None

	def voice(self, number):
		voices = self.speaker.getProperty("voices")
		self.speaker.setProperty("voice", voices[number].id)

	def say(self, stringInput):
		self.voice(self.voiceID)
		self.speaker.say(stringInput)
		self.speaker.runAndWait()

	def getFields(self, typeField):
		if typeField == "news":
			return ["news", "get news", "read news", "hot news"]

		if typeField == "wikipedia":
			return ["wikipedia", "wiki", "more detail"]

	def earing(self):
		with sr.Microphone() as mic:
			print("Robot: I'm listening...")
			#audio = self.recognizer.listen(mic)
			audio = self.recognizer.record(mic, duration=5)
		try:
			text = self.recognizer.recognize_google(audio, language = self.record_language)
			#text = self.recognizer.recognize_google(audio)
		except:
			text = self.none
		return text

	def helloWorld(self):
		stringInput = "Hi, I'm " + self.name + ". Can I help you?"
		self.say(stringInput)

	def goodBye(self):
		stringInput = "Goodbye, see you again!"
		self.say(stringInput)

	#module
	def readNews(self, title):
	    queue = title
	    params = {
	        'apiKey': '30d02d187f7140faacf9ccd27a1441ad',
	        "q": queue,
	    }

	    api_result = requests.get('http://newsapi.org/v2/top-headlines?', params)
	    api_response = api_result.json()

	    if api_response['status'] == "error" or len(api_response['articles']) < 1:
	    	self.say("Not found anything about " + title)

	    for number, result in enumerate(api_response['articles'], start=1):
	        print(f"""Tin {number}:\nTiêu đề: {result['title']}\nTrích dẫn: {result['description']}\nLink: {result['url']}
	    """)
	        if number <= 3:
	            webbrowser.open(result['url'])

	def openWebsite(self, text):
		reg_ex = re.search('website (.+)', text)
		if reg_ex:
			domain = reg_ex.group(1)
			url = 'https://www.' + domain
			webbrowser.open(url)
			self.say("your request website opened.")

	def handleActionKeyWork(self, action, sentence, confirm = None):
		switcher = {
	        "voice": "this is voice",
	        "website": self.openWebsite(sentence),
	    }

		func = switcher.get(action)
		return func

	def wikipediaSummary(self, keywords):
		self.wikipedia.set_lang('en')
		if keywords != "":
			contents = self.wikipedia.summary(keywords)
			if len(contents) > 0:
				print(contents[0])
				#self.say(contents[0])

	def contentSearch(self, sentence):
		keywords = sentence.split("about")
		if len(keywords) > 1 and keywords[1] != "":
			return keywords[1]
		return ""

	#Helper func
	def getStopWords(self, language):
		return stopwords.words(language)

	def tokenize(self, sentence):
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(sentence)
		return tokens
		#return self.nltk.word_tokenize(sentence)

	def cleanTokens(self, sentence, language):
		sw = self.getStopWords(language)
		firstTokens = self.tokenize(sentence)
		clean_tokens = [token for token in firstTokens if token not in sw]
		return clean_tokens


	def taggingText(self, sentence):
		tokens = self.cleanTokens(sentence, "english")
		return self.nltk.pos_tag(tokens)

	def isExistKeyword(self, sentence, fields):
		for keyword in fields:
			if sentence.find(keyword) >= 0:
				return True
		return False

	#training by Sequential model
	def loadDataJson(self, file):
		with open(file) as file:
			self.data = json.load(file)

		for intent in self.data['intents']:
		    for pattern in intent['patterns']:
		        self.training_sentences.append(pattern)
		        self.training_labels.append(intent['tag'])
		    self.responses.append(intent['responses'])
		    
		    if intent['tag'] not in self.labels:
		        self.labels.append(intent['tag'])

		num_classes = len(self.labels)

	def training(self):
		#use “LabelEncoder()” function provided by scikit-learn to convert the target labels into a model understandable form
		lbl_encoder = LabelEncoder()
		lbl_encoder.fit(self.training_labels)
		self.training_labels = lbl_encoder.transform(self.training_labels)

		vocab_size = 1000
		embedding_dim = 16
		max_len = 20
		oov_token = "<OOV>"

		tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
		tokenizer.fit_on_texts(self.training_sentences)
		word_index = tokenizer.word_index
		sequences = tokenizer.texts_to_sequences(self.training_sentences)
		padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

		#Model training
		model = Sequential()
		model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
		model.add(GlobalAveragePooling1D())
		model.add(Dense(16, activation='relu'))
		model.add(Dense(16, activation='relu'))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss='sparse_categorical_crossentropy', 
		              optimizer='adam', metrics=['accuracy'])

		model.summary()
		history = model.fit(padded_sequences, np.array(self.training_labels), epochs=500)
		model.save("chat_model")

		# to save the fitted tokenizer
		with open('tokenizer.pickle', 'wb') as handle:
		    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		    
		# to save the fitted label encoder
		with open('label_encoder.pickle', 'wb') as ecn_file:
		    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

		self.loadModel()

	def loadModel(self):
		# load trained model
	    self.model = keras.models.load_model('chat_model')

	    # load tokenizer object
	    with open('tokenizer.pickle', 'rb') as handle:
	        self.tokenizer = pickle.load(handle)

	    # load label encoder object
	    with open('label_encoder.pickle', 'rb') as enc:
	        self.lbl_encoder = pickle.load(enc)

	    # parameters
	    self.max_len = 20

	def analyzerRespone(self, inputText):
		result = self.model.predict(keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([inputText]), truncating='post', maxlen=self.max_len))
		tag = self.lbl_encoder.inverse_transform([np.argmax(result)])
		for i in self.data['intents']:
			if i['tag'] == tag:
				return (np.random.choice(i['responses']))


	def init(self):
		inputText = self.earing()
		if inputText == "shut down" or inputText == "shutdown" or inputText == "stop" or inputText == "quit":
			print("system stopped.")
			self.goodBye()
			self.say("system stopped")
			quit()
		else:
			if inputText == self.none:
				self.init()
			else:
				print("You: ", inputText)
				response = self.analyzerRespone(inputText)
				if response != "":
					print("Robot: " + response)
					self.say(response)
				
				# inputText = inputText.lower()
				# self.sentenceAnalyzer(inputText)
				
				self.init()


def main():
	#robot = Robot(nltk, 'vi-VN')
	record_language = 'en-EN'
	robot = Robot(nltk, wikipedia, record_language)
	robot.loadDataJson("intents.json")
	robot.loadModel()
	robot.helloWorld()
	robot.init()

main()




