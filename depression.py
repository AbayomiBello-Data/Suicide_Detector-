import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer


st.set_page_config(layout="wide")

# -----------------------------------------------------------------

st.markdown("""
<style>
div.stButton > button:first-child {
background-color: #00cc00;
color:black;
font-size:15px;
height:2.7em;
width:20em;
border-radius:10px 10px 10px 10px;}
</style>
    """,
            unsafe_allow_html=True
            )

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTI8DRxgEP4PMaChfWJQKulfwMWdF486bB0SF0ZHXkgS5z4gc2Jd7EGKC8-gjjKWNxEUlQ&usqp=CAU")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.image(r"https://tse4.mm.bing.net/th?id=OIP.a5cp6HV0bAKzC2UYiCl0dgHaEt&pid=Api&P=0", width=500)




# Load the saved model
with open('logistic_re.pkl', 'rb') as file:
    classifier, vectorizer = pickle.load(file)

# Define a Streamlit app
def app():
    st.title("Suicide Risk Prediction")
    user_input = st.text_input("Please enter the text:")
    if st.button('Predict'):
        # Vectorize the input text
        text_vectorized = vectorizer.transform([user_input])

        # Make the prediction using the trained model
        prediction = classifier.predict(text_vectorized)[0]

        # Set the color of the prediction text
        color = "red" if prediction == 'suicide' else "green"

        # Display the prediction result with the specified color
        if color == "red":
            st.write('The model predicts that the writer of this text is likely depressed and has a high risk of suicide.')
        else:
            st.write('The model predicts that the writer of this text is not depressed and has no intention of suicide.')

        # Display the prediction result with the specified color
        st.markdown(f'<p style="color:{color};">{prediction}</p>', unsafe_allow_html= True)

if __name__ == '__main__':
    app()
