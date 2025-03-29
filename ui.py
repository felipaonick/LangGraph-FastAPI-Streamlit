import streamlit as st
import requests

st.set_page_config(
    page_title="LangGraph Agent UI",
    page_icon=":robot_face:",
    layout="centered",
)

# define API endpoint
API_URL = "http://127.0.0.1:8000/chat"


MODEL_NAMES = [
    "qwen2.5:14b",
    "qwen2.5:7b"
]

# STREAMLIT UI Elements
st.title("LangGraph ChatBot Agent")
st.write("Interact with the LangGraph-besed agent using this interface.")


# Input box for system prompt
given_system_prompt = st.text_area("Define you AI Agent:", height=100, placeholder="Type your system prompt here...")

# Dropdown for selecting the model
select_model = st.selectbox("Select Model:", MODEL_NAMES)

# INput box for user message
user_input = st.text_area("Enter your message(s):", height=150, placeholder="Type your message heere...")

# Button to send the query
if st.button("Send Query"):
    if user_input.strip():
        try:
            # Send the input to the FastAPI backend
            payload = {"messages": [user_input], "model_name": select_model, "system_prompt": given_system_prompt}
            response = requests.post(API_URL, json=payload)
            
            # Display the response
            if response.status_code == 200:
                response_data = response.json()
                if "errors" in response_data:
                    st.error(f"Error: {response_data['errors']}")
                else:
                    ai_responses = [
                        message.get("content", "")
                        for message in response_data.get("messages", [])
                        if message.get("type") == "ai"
                    ]

                    if ai_responses:
                        st.subheader("Agent Response:")
                        st.markdown(f"**Final Response:** {ai_responses[-1]}")
                    else:
                        st.warning("No AI response found in the agent output.")
            else:
                st.error(f"Request failed with status code {response.status_code}.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a message before cliking 'Send Query'.")