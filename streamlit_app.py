import streamlit as st
from phi.assistant import Assistant
from phi.llm.groq import Groq
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pymysql  # Changed to pymysql
import os

# Load environment variables
load_dotenv()

# Function to create MySQL connection
def create_mysql_connection():
    try:
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )
        return conn
    except pymysql.MySQLError as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Initialize the assistant
@st.cache_resource
def get_assistant():
    return Assistant(
        llm=Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("Groq_API_key")),
        description="I am a helpful AI assistant powered by Groq. How can I assist you today?",
    )

# Function to fetch data from MySQL
def fetch_trainer_data():
    conn = create_mysql_connection()
    if conn is None:
        return []  # Return an empty list if connection failed
    try:
        cursor = conn.cursor()
        # Fetching id, name, domain, skills, experience, and the chatroom link for each trainer
        cursor.execute("SELECT id, name, domain, skills, experience, links FROM person")  # Adjusted to match new table structure
        records = cursor.fetchall()
        return records
    except pymysql.MySQLError as e:
        st.error(f"Error fetching data: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# Function to generate embeddings using SentenceTransformer
@st.cache_resource
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    ids = []
    
    for record in data:
        person_id, name, domain, skills, experience = record[:5]  # Adjusted for the new column
        description = f"{name}, {experience} years of experience, skilled in {skills}, and works in {domain}."
        embedding = model.encode(description)
        ids.append(person_id)
        embeddings.append(embedding)
    
    return ids, np.array(embeddings)

# Function to build FAISS index
@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance is used for similarity
    index.add(embeddings)
    return index

# Function to perform FAISS search
def search_in_faiss(index, query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

# Function to create a clickable button for each trainer
def create_trainer_button(trainer_name, link):
    # This will create a button with the text 'Chat with [trainer_name]' and redirect to their chat link when clicked
    st.markdown(f'<a href="{link}" target="_blank"><button style="background-color:#4CAF50; color:white; padding:10px; border:none; border-radius:4px; cursor:pointer;">Chat with {trainer_name}</button></a>', unsafe_allow_html=True)

    

# Streamlit app
st.title("Chat with Groq-powered Assistant")

# Input text box for user question
user_input = st.text_input("What do you want to know?", "")

# Generate embedding for the user query outside the button click logic
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([user_input]) if user_input.strip() else None  # Avoid encoding if input is empty

# Button to get the response
if st.button("Ask"):
    if user_input.strip():  # Check for non-empty input
        with st.spinner("Generating response..."):
            # Get the response from Groq-powered assistant
            response_generator = get_assistant().chat(user_input)
            response = "".join([chunk for chunk in response_generator if isinstance(chunk, str)])
            st.markdown(response)

            # Fetch trainer data from MySQL
            trainer_data = fetch_trainer_data()

            # Generate embeddings from database data
            ids, embeddings = generate_embeddings(trainer_data)

            # Build FAISS index with embeddings
            faiss_index = build_faiss_index(embeddings)

            # Perform FAISS search for similar results
            distances, indices = search_in_faiss(faiss_index, np.array(query_embedding))

            # Display matched results from database
            st.markdown("### Related Trainers from Database:")
            for idx in indices[0]:
                # Fetch trainer details including link from the result
                person_id, name, domain, skills, experience, link = trainer_data[idx]  # Adjusted for the new column structure
                
                # Display trainer info
                st.markdown(f"- **Name**: {name}, **Domain**: {domain}, **Skills**: {skills}, **Experience**: {experience} years")
                
                # Create clickable button to chat with trainer
                create_trainer_button(name, link)  # Pass the link to the button creation function
    else:
        st.warning("Please enter a question.")
