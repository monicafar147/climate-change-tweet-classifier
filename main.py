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
	st.subheader("Classifing tweets towards their belief in Climate Change")
	from PIL import Image
	logo = Image.open('resources/imgs/twitter_logo.jpg')
	st.image(logo, width = 100)

	options = ["Home", "About","Data Cleaning","Modelling"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the predication page
	if selection == "Home":
		st.write('-----------------------------------------------')
		st.info("See the **About** page for more information and data analysis")

		# Creating a text box for user input
		st.markdown("## **Enter Tweet Below**")
		tweet_text = st.text_area("","Type Here")

		'''if st.button("Classify Linear SVC model"):
			tweet_pred = svc.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))

		if st.button("Classify SVC (gridsearch) model"):
			tweet_pred = grid.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))

		if st.button("Classify LR model"):
			tweet_pred = lr.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))'''

	if selection == "About":
		# Title
		st.title('About')
		st.write('-----------------------------------------------')
		# st.info("General Information")

		# Intro
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

		# EDA
		st.markdown('## Exploratory Data Analysis')
		st.subheader('Most tweeted hashtag')
		st.info("""\n anti : #MAGA (11) 
				\n neutral : #climate (16)
				\n pro : #climate : (130)
				\n news : #climate : (130)
				""")
		st.subheader('Most tweeted username')
		st.info("""\n anti : @realDonaldTrump (71)
				\n neutral : @StephenSchlegel (307)
				\n pro : @realDonaldTrump (31)
				\n news : @thehill (77)
				""")

		st.subheader('Interesting insights about the training data: ')
		st.markdown("""For Anti-climate change tweets:""")
		st.info("""\n - The word **science** occurs alot as many people against climate change
				use scientific facts to express their view.
				\n - **Steves Goddard** pops up and he is a social media influencer of
				 scientific background but denies science about climate change. 
				He would often produce a graphic that shows that climate change is not real.
				\n - Politicians referenced include **Al Gore**, **Obama** and **Donald Trump**.
				\n - The **#MAGA** is a hasthtag that means Make America Great Again and
				 is associated with Americans who side with Trump.""")
		st.markdown("""For Pro-climate change tweets:""")
		st.info("""\n - **Steven Schlegel** occurs often as there is a science report
				from 2006 detailing the predictions of climate change in the future published by Schlegel.
				\n - The hashtag **#ImVotingBecause** occured 62 times which might imply that climate change
				can influence voters decision who to vote for.
				\n - The word husband pops up for some reason.
				\n - Politicians referenced include **Sen Sanders** and **Donald Trump**""")
		st.markdown("""For Neutral tweets:""")
		st.info("""\n - The word **journalist** pops up.
	\n - Places referrenced are **America** and **Paris**. The paris word might reference to *The Paris Agreement* 
	\n - **Chelsea Clinton** is referrenced, she is an American author and global health advocate.
	\n - Politicians referenced include **Sen Sanders** and **Donald Trump**
	\n - Celebrities referenced include **Leonardo Dicaprio**.
	\n - Strong emotional words include please, action, fuck and responsible""")
		st.markdown("""For News tweets:""")
		st.info("""\n - The word **EPA** pops up, this is the United States Environmental Protection Agency.
	\n - News outlets referenced include **CNN**, **Guardian**, **Time**.
	\n - **Scott Prutt** is mentioned, he is the Administrator of the United States Environmental Protection Agency.
	\n - The word **independent study** pops up.
	\n - Che **white house** and **Trump** is mentioned.
	\n - Countries that pop up include **US** and **China**.""")

	if selection == "Data Cleaning":
		st.write('-----------------------------------------------')
		st.write("""Cleaning tweets is vitally important for an accurate model. 
				\n Try our tweet cleaner below that will show you how we cleaned our tweets.""")
		st.markdown("## **Enter Tweet Below**")
		# pickle preprocessing function
		'''process_path = "resources/process.pkl"
		with open(process_path,'rb') as file:
			process = pickle.load(file)
		tweet_text = st.text_area("","Type Here")
		tweet_processed = process(tweet_text)
		if st.button("Clean Tweet"):
			st.success("Tweet cleaned as: {}".format(tweet_processed))'''
		st.subheader("Before preprocessing we need to ask ourseleves the following questions about this data:")
		st.write("Does URL have impact on the tweet sentiment?")
		st.info("""So many twitter users retweet URL's to substantiate their view, therefore by removing 
				the URL the sentiment value might be reduced.""")
		st.write("Does retweet have any impact on tweet sentiment?")
		st.info("""\n Number of Original Tweets: 6133
				\n Number of Retweets: 9687
				\n Ratio of Orignal Tweets to retweets: 0.63
				\n Because the retweet ratio is more than 0.5, it would be better to keep the retweet as RT.""")
		st.write("Does removing hashtags remove sentiment?")
		st.info("""Hashtags can link one tweet to another, therefore it would be better to keep the hashtags.""")
		st.write("Contractions are a problem. how will removing them effect our model?")
		st.info("""Twitter users use slang to communicate their views and many tweets contain contractions. 
				Contractions does make the modelling process more challenging as [don't] needs to mean the same
				as [do not]. 
				\n Using the TwitterTokenizer module helps to keep contractions in their own form.""")
		st.write("What about emojis?")
		st.info("""Twitter users love using emojis as a way of expressing their emotions. The training data did not
				have a large sample of emojis so keeping the emojis in their raw form would improve the sentiment
				of our models.""")

	if selection == "Modelling":
		st.write('-----------------------------------------------')
    
if __name__ == '__main__':
    
    main()
