import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()


try:

    model = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key) # Keeping the original model

    generic_template = """
    Please answer the user's question thoroughly in {language}.
    Craft a response that is descriptive, providing ample detail, context, and background information.
    At the same time, ensure the answer is clear, easy to understand (using simple language),
    and structured logically to make the information easy to remember.
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", generic_template), ("user", "{text}")]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

except Exception as e:
    st.error(f"Failed to initialize the language model: {e}")
    st.stop()


# --- Streamlit App ---
st.title("Language Personalized Chat")

if "language" not in st.session_state:
    st.session_state.language = None # Store the selected language
if "messages" not in st.session_state:
    st.session_state.messages = [] 

# --- Language Selection (Only if not already set) ---
if st.session_state.language is None:
    st.header("1. Select Your Preferred Language")
    available_languages = ("Bengali", "English", "Hindi", "Urdu", "French", "Spanish")
    selected_lang = st.selectbox(
        "Choose the language for the AI's responses:",
        options=available_languages,
        index=1 
    )

    if st.button("Confirm Language"):
        st.session_state.language = selected_lang
        welcome_message = {
            "Bengali": "স্বাগত! আপনার প্রশ্ন জিজ্ঞাসা করুন।",
            "English": "Welcome! Ask your questions below.",
            "Hindi": "स्वागत है! अपने प्रश्न पूछें।",
            "Urdu": "خوش آمدید! اپنے سوالات پوچھیں۔",
            "French": "Bienvenue ! Posez vos questions ci-dessous.",
            "Spanish": "¡Bienvenido! Haga sus preguntas a continuación."
        }.get(selected_lang, f"Welcome! Ask your questions in {selected_lang}.")

        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        st.rerun() # Rerun the script to move to the chat interface

else:
    st.header(f"2. Chat in {st.session_state.language}")
    st.write("Ask multiple questions below. The AI will respond in your selected language.")
    st.markdown("---")


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if user_question := st.chat_input("What question do you have in mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_question)

        # --- Generate and Display AI Response ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Create a placeholder for the streaming response
            full_response = ""
            try:
                # Stream the response from the chain
                # Use st.write_stream for cleaner streaming display
                stream_input = {"language": st.session_state.language, "text": user_question}
                full_response = st.write_stream(chain.stream(stream_input))



            except Exception as e:
                st.error(f"An error occurred while getting the response: {e}")
                full_response = "Sorry, I encountered an error."
                message_placeholder.markdown(full_response) # Update placeholder with error

        # Add AI response to chat history *after* it's fully generated
        if full_response: # Ensure we don't add empty responses on error
             st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Note: No explicit st.rerun() needed here, as st.chat_input triggers a rerun on submission.

    # Add a button to reset the session (optional)
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Chat and Change Language"):
        # Clear relevant session state keys
        st.session_state.language = None
        st.session_state.messages = []
        st.rerun() # Rerun to go back to language selection