#%%
import pandas as pd
import numpy as np
import streamlit as st
from streamlit import session_state as session
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import pickle



df = pd.read_csv("Data/clean_data.csv")





#%%
#Import TfIdfVectorizer from the scikit-learn library

#step1: select features 
features_df = ['review_score','out_of', 'city', 'food_type', 'food_type1','location', 'number_of_reviews', 'opening_hour']

for feature in features_df:
    df[feature] = df[feature].fillna('')

#step2: combine features 
def comine_features(row):
    try:
        return row['review_score']+ " "+row['out_of'] + " "+row['city']+" "+row['food_type'] + " "+row['food_type1'] + " "+row['location']+" "+row['number_of_reviews']+" "+row['opening_hour']
    except:
        print("Error:", row)

df['combined_features'] = df.apply(comine_features, axis=1)

    





# Compute the cosine similarity matrix

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df.index, index=df['restaurant_name'])


def content_recommender(restaurant_name, cosine_sim=cosine_sim, df=df, indices=indices):
    # Obtain the index of the movie that matches the title

    idx = indices[restaurant_name]

    # Get the pairwsie similarity scores of all restaurants with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the restaurants based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar restaurants. Ignore the first movie.
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    restaurant_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar restaurants
    return df['restaurant_name'].iloc[restaurant_indices] 


pickle.dump(df.to_dict(),open('Data/rest.pkl','wb'))



rest_dict = pickle.load(open('Data/rest.pkl','rb'))
restaurants = pd.DataFrame(rest_dict)


#%%
dataframe =None
st.title("""
Restaurants Recommendation System
 """)


st.text("")
st.text("")
st.text("")
st.text("")

session.options = st.selectbox(label="Select restaurant", options= restaurants.restaurant_name)

st.text("")
st.text("")
st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

is_clicked = col1.button(label="Recommend")

if is_clicked:
    dataframe = content_recommender(session.options)

st.text("")
st.text("")
st.text("")
st.text("")

if dataframe is not None:
    st.table(dataframe)



# %%
