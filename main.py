"""
    Streamlit web developer application to classify tweets from twitter.
    Author: Monica Farrell.
"""
# Streamlit dependencies
import streamlit as st

# general
import numpy as np 
import pandas as pd
import dill as pickle

# text preprocessing
import re
from string import punctuation
import nltk
nltk.download(['stopwords','punkt'])
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.write('-----------------------------------------------')
	st.title("Climate Change Tweet Classifier")

	options = ["Home", "About","Data Cleaning","Modelling"]
	selection = st.sidebar.selectbox("Pages", options)
	# pickle preprocessing function
	process_path = "resources/process.pkl"
	with open(process_path,'rb') as file:
		process = pickle.load(file)

	# loading grid search best fit model
	model_load_path = "resources/linear_SVC.pkl"    
	with open(model_load_path,'rb') as file:
		grid = pickle.load(file)

	result = {-1:'Anti Climate Change',
			0: 'Neutral towards Climate Change',
			1: 'Pro Climate Change',
			2: 'News'}

	# Building out the predication page
	if selection == "Home":
		from PIL import Image
		logo = Image.open('images/twitter_logo.jpg')
		st.image(logo, width = 100)
		st.write('-----------------------------------------------')
		st.info("See the **About** page for more information and data analysis")

		# Creating a text box for user input
		st.markdown("## **Enter Tweet Below**")
		tweet_text = st.text_area("","Type Here")
		tweet_processed = process(tweet_text)

		if st.button("Classify"):
			tweet_pred = grid.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))

	if selection == "About":
		st.write('-----------------------------------------------')
		option = st.sidebar.radio(
			"Data analysis",
			("Introduction",
			"Anti climate change",
             "Neutral towards climate change",
             "Pro climate change",
			 "News"
				)
			)
		if option == "Introduction":
			st.markdown('## Introduction')
			st.info("""Climate change has been a trending topic ever since
				 Al Gore received a Nobel Peace Prize for his campaign in 2007.
				The topic has become a controversial subject on twitter where some 
				twitter users feel very strongly that climate change is not real 
				and is part of a conspiracy theory. To add fire to the situation, 
				American President, Donald Trump, claimed that climate change is a 
				Chinese-funded conspiracy. As a result, some twitter users
				started tweeting that Climate Change is not real and trying to
				follow tweets about climate change suddenly required a degree in politics.""")

		if option == "Anti climate change":
			st.write('info')

		if option == "Neutral towards climate change":
			st.write('info')
		if option == "Pro climate change":
			st.write('info')
		if option == "News":
			st.write('info')

	if selection == "Data Cleaning":
		option = st.sidebar.radio(
			"Before preprocessing we need to ask ourseleves the following questions about this data:",
			("URL",
			"Retweets",
			"hashtags",
             "Language",
			 "Emojis"))				
		st.write('-----------------------------------------------')
		st.write("""Cleaning tweets is vitally important for an accurate model. 
				\n Try our tweet cleaner below that will show you how we cleaned our tweets.""")
		st.markdown("## **Enter Tweet Below**")
		# pickle preprocessing function
		tweet_text = st.text_area("","Type Here")
		tweet_processed = process(tweet_text)

		if st.button("Clean Tweet"):
			st.success("Tweet cleaned as: {}".format(tweet_processed))

		st.subheader("Before preprocessing we need to ask ourseleves the following questions about this data:")

		if option == "URL":
			st.markdown('## Does removing the URL shared in posts have an impact?')
			st.info("""So many twitter users retweet URL's to substantiate their view, therefore by removing 
				the URL the sentiment value might be reduced.""")

		if option == "Retweets":
			st.markdown('## Does retweet have any impact on tweet sentiment?')
			st.info("""\n Number of Original Tweets: 6133
				\n Number of Retweets: 9687
				\n Ratio of Orignal Tweets to retweets: 0.63
				\n Because the retweet ratio is more than 0.5, it would be better to keep the retweet as RT.""")

		if option == "hashtags":
			st.markdown('## Does removing hashtags remove sentiment?')
			st.info("""Hashtags can link one tweet to another, therefore it would be better to keep the hashtags.""")

		if option == "Language":
			st.markdown('## Contractions are a problem. how will removing them effect our model?')
			st.info("""	Twitter users use slang to communicate their views and many tweets contain contractions. 
				Contractions does make the modelling process more challenging as [don't] needs to mean the same
				as [do not]. 
				\n Using the TwitterTokenizer module helps to keep contractions in their own form.""")

		if option == "Emojis":
			st.markdown('## What about emojis?')
			st.info("""Twitter users love using emojis as a way of expressing their emotions. The training data did not
				have a large sample of emojis so keeping the emojis in their raw form would improve the sentiment
				of our models.""")



	if selection == "Modelling":
		st.write('-----------------------------------------------')
    
if __name__ == '__main__':
    
    main()
