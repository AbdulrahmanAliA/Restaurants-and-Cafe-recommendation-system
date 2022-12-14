# Restaurants and Cafe recommendation system
## Introduction 
A restaurant is a business formation which prepares and serves food and drink to customers in return for money

Restaurants are essential sites where food, family, and friends interact, in favorite and good restaurants people can feel comfortable and rest and spend a lot of time enjoying a good restaurant.

We want to help users to make a choice. 

## Dataset
The data for this project will be read into data frame (Find the dataset on the following [link](https://www.kaggle.com/datasets/norahalsharif/saudiarabia-restorations)).
This dataset was scraped from TripAdvisor Tripadvisor, the world's largest travel platform. It produces a lot of restaurants each one of them has his one value wither it was in the type of food or the price and locations.

This data contains information about restaurants in 3 main cities in Saudi Arabia: JEDDAH , RYADH, Eastern Province.

## Recommender
for this project i will use Content Based Filtering:

tries to guess the features or behavior of a user given the item’s features, he/she reacts positively to.

Feature used :  review score, out of, city, food type, food type1, location, number of reviews, opening hour

for calculating similarity we will use Cosine similarity which is a metric used to measure how similar the documents are irrespective of their size

TF-IDF and count vectorizer both are methods for converting text data into vectors as model can process only numerical data.

## Tools
- Python
- Numpy
- Pandas
- Sklearn
- Matplotlib
- Pickle
- Streamlit

## References

https://docs.streamlit.io/library/get-started

https://towardsdatascience.com/build-a-song-recommendation-system-using-streamlit-and-deploy-on-heroku-375a57ce5e85

https://www.quora.com/What-is-the-difference-between-TfidfVectorizer-and-CountVectorizer-1

https://www.datacamp.com/tutorial/recommender-systems-python



