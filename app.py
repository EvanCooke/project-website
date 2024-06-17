import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np
from keras.layers import TFSMLayer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def download_nltk_resources():
    resources = ["punkt", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

# Download necessary NLTK resources at the start of your app
download_nltk_resources()

# load sms spam classifier model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
# from scipy.sparse import csr_matrix
# tfidf = csr_matrix(pickle.load(open('vectorizer.pkl','rb')))
# model = csr_matrix(pickle.load(open('model.pkl','rb')))

# load handwritten model
@st.cache_resource
def load_model():
    handwritten_model = tf.saved_model.load('handwritten.model')
    return handwritten_model

handwritten_model = load_model()

# load the sentiment classifier model
sentiment_classifier_model = pickle.load(open('simple-sentiment-classifier-model.pkl', 'rb'))

st.title("My Machine Learning Projects")

# Add a horizontal line
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)

# Section 5: Sentiment Classifier
st.markdown("## Simple Sentiment Classifier")
st.markdown("Enter the message you want to classify as happy, sad, or neutral.")

input_sentiment_msg = st.text_area("", key=5)

if st.button('Predict Sentiment', key=3):

    # 1. predict
    sentiment_result = sentiment_classifier_model.predict([input_sentiment_msg])[0]
    # 2. Display
    if sentiment_result == 1:
        st.header("**Result: Negative**")
    elif sentiment_result == 2:
        st.header("**Result: Neutral**")
    elif sentiment_result == 3:
        st.header("**Result: Positive**")
    else:
        st.header("**Error, try again.**")

# Add a horizontal line
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)

# Section 2: Handwritten Digit Identifier
st.markdown("## Handwritten Digit Identifier")
st.markdown("Draw a digit in the box below and click 'Identify' to predict the digit.")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
    stroke_width=20,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

#Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#     rescaled = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
#     st.write("Rescaled image")
#     st.image(rescaled)

if st.button('Identify Digit', key=2):
    # img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = canvas_result.image_data.astype('uint8') # Get the image data from the canvas (this is already a NumPy array)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    img = cv2.resize(img, (28, 28)) # Resize the image to 28x28 pixels
    img = img / 255.0 # Normalize the pixel values to [0, 1]
    img = 1 - img # Invert the normalized image to get white digit on black background
    img = img.astype('float32') # Convert the image data to float32

    img = img.reshape(1, 28, 28) # Reshape the image data for prediction
    
    # Predict the digit
    # val = handwritten_model.predict(img, training=False)
    val = handwritten_model(img, training=False)

    
    st.write(f"**Result: {np.argmax(val[0])}**")

# Add a horizontal line
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)

# Section 3: Snake AI
st.markdown("## Deep Q Learning Snake AI")

# Display the image
#st.image("snake-ai-picture.PNG")
st.image("snake-ai-gif.gif")

st.markdown("<a href='https://github.com/EvanCooke/deep-q-learning-snake-ai'>Source Code</a>", unsafe_allow_html=True)

# Add a horizontal line
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)

# Section 4: Buyer Classifier
text = """
Using Python and Scikit-Learn's logistic regression algorithm to develop a classification model,
this model predicts whether a visitor to an online store will make a purchase or not, providing valuable insights into customer behavior. 
By understanding the circumstances under which customers make purchase decisions, businesses can tailor their marketing strategies more effectively. 
"""

st.markdown("## Implemented a logistic regression model to predict the purchase intentions of online shoppers")
st.markdown(text)

st.markdown('### Model Evaluation:')
st.image('buyer-classifier-model-stats.PNG')
st.markdown('### Feature Importance:')
st.image("feature-importance.PNG")
st.markdown('### Feature Correlation Heatmap:')
st.image("heatmap.PNG")

# Add a horizontal line
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)


# Section 1: Email/SMS Spam Classifier
st.markdown("## Email/SMS Spam Classifier")
st.markdown("Enter the message you want to classify as spam or not spam.")

input_sms = st.text_area("", key=4)

if st.button('Predict Spam', key=1):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("**Result: Spam**")
    else:
        st.header("**Result: Not Spam**")

# Add a horizontal line
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)

st.markdown("GitHub: <a href='https://github.com/EvanCooke'>github.com/EvanCooke</a>", unsafe_allow_html=True)
st.markdown("Email: <a href='mailto:evan-cooke@uiowa.edu'>evan-cooke@uiowa.edu</a>", unsafe_allow_html=True)
