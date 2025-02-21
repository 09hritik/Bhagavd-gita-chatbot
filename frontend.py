import requests
import streamlit as st

API_URL = "http://localhost:8000/answer"

def get_chat_response(input_text):
    try:
        response = requests.post(API_URL, json={"input": input_text})
        response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
        
        response_json = response.json()  # Try parsing JSON
        return response_json.get('answer', 'No answer found')

    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to API. {str(e)}"
    except ValueError:
        return "Error: Invalid JSON response from server."

st.title("Gita AI : Powered by Bhagavad Gita and OpenAI")
input_text = st.text_input("Write your question")

if input_text:
    st.write(get_chat_response(input_text))