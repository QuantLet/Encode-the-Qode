import streamlit as st
from aitextgen import aitextgen

DEFAULT_Q = "https://github.com/QuantLet/Encode-the-Qode/blob/main/Covert-Repos-To-DF_deprecate/parse_repos_to_df.ipynb"
DEFAULT_Q = DEFAULT_Q.replace('https://github.com/', 'https://raw.githubusercontent.com/')

q2key_model = aitextgen()

Q_INPUT = st.text_input(  
    label="Enter your Quantlet Link here", 
    value=DEFAULT_Q,
)
BUTTON_STATE = st.button("Submit")
print(BUTTON_STATE)

if BUTTON_STATE:
    generated_text = q2key_model.generate_one(Q_INPUT, max_length=10, temperature=0.7, top_p=0.9,)

st.text(generated_text)