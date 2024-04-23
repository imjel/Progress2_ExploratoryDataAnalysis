import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('best_rf_model-2.pkl')

# Title of the web app
st.title('Spotify & Youtube Hit Classifier - Random Forest Model Deployment')
st.subheader('Robin Tran, Max Endieveri, Marie Nguyen')

st.image('1684909865437.png', caption='Spotify & Youtube', use_column_width=True)

# Input fields for the features your model uses
Danceability = st.number_input('Enter Danceability')
Energy = st.number_input('Enter Energy')
Loudness = st.number_input('Enter Loudness')
Key = st.number_input('Enter Key')
Speechiness = st.number_input('Enter Speechiness')
Instrumentalness = st.number_input('Enter Instrumentalness')
Acousticness = st.number_input('Enter Acousticness')
Liveness = st.number_input('Enter Liveness')
Valence = st.number_input('Enter Valence')
Tempo = st.number_input('Enter Tempo')
Duration_ms = st.number_input('Enter Duration in milliseconds')

# Button to make predictions
if st.button('Predict'):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[Danceability, Energy, Loudness, Key, Speechiness, Instrumentalness, Acousticness, Liveness, Valence, Tempo, Duration_ms]], columns=['Danceability', 'Energy', 'Loudness', 'Key','Speechiness', 'Instrumentalness', 'Acousticness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms'])
    # Use the model to make predictions
    prediction = model.predict(input_data)
    prediction_result = prediction[0]  # This captures the single prediction result

    # Use conditional statements to determine the output message
    if prediction_result == 0:
        message = "Not a hit :("
    elif prediction_result == 1:
        message = "Semi-hit. Has potential to become a hit"
    elif prediction_result == 2:
        message = "It is a hit!"

    # Display the appropriate message based on the prediction
    st.write(f'Prediction: {message}')
        
