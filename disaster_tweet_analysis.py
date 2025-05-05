import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional
import json
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv(override = True)

# Pydantic model for structured output
class DisasterInfo(BaseModel):
    number_of_people: Optional[int] = Field(
        default=None, description="Estimated number of people affected, if mentioned"
    )
    location: Optional[str] = Field(
        default=None, description="Location of the disaster, if mentioned"
    )
    severity: Optional[str] = Field(
        default=None, description="Severity level (e.g., low, medium, high, critical)"
    )
    important_notes: Optional[str] = Field(
        default=None, description="Any important notes or additional information like pregnancy status,missing persons, etc."
    )

# Initialize Google Gemini model with LangChain
# Make sure to set GOOGLE_API_KEY in your .env file
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    max_output_tokens=1024,
    google_api_key=os.getenv("GOOGLE_API_KEY")
).with_structured_output(DisasterInfo)

# Prompt template for tweet analysis
prompt_template = """
You are an expert in disaster response analysis. Analyze the following tweet and extract structured information about the disaster, including the number of people affected, the location, and the severity (low, medium, high, or critical). If any information is missing or unclear, return null for that field. Return the output in JSON format.

Tweet: {tweet}

Instructions:
- Number of people: Extract an estimated number of affected people, if mentioned (e.g., "100 people trapped" → 100). If not mentioned, return null.
- Location: Identify the specific location (e.g., city, region) if mentioned. If not mentioned, return null.
- Severity: Assess the severity based on the tweet's content (e.g., "minor flooding" → low, "massive earthquake" → critical). If unclear, return null.
- Important notes: Include any important notes or additional information like pregnancy status, missing persons, etc. If not mentioned, return null."""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create LangChain pipeline
chain = prompt | llm 

# Streamlit interface
st.title("Disaster Tweet Analysis")
st.write("Enter a tweet to analyze disaster-related information using Google Gemini and LangChain.")

# Input field for tweet
tweet_input = st.text_area("Tweet Text", placeholder="e.g., Earthquake in San Francisco, 50 people injured, major damage reported")

# Button to analyze
if st.button("Analyze Tweet"):
    if tweet_input.strip():
        try:
            with st.spinner("Analyzing tweet..."):
                # Invoke the chain with the tweet input
                result = chain.invoke({"tweet": tweet_input})
                
                # Display results
                st.subheader("Analysis Results")
                st.json(result.model_dump_json())
                
                # Display formatted output
                st.write("**Extracted Information:**")
                st.write(f"- **Number of People Affected**: {result.number_of_people or 'Not mentioned'}")
                st.write(f"- **Location**: {result.location or 'Not mentioned'}")
                st.write(f"- **Severity**: {result.severity or 'Not mentioned'}")
                st.write(f"- **Important Notes**: {result.important_notes or 'Not mentioned'}")
        except Exception as e:
            st.error(f"Error analyzing tweet: {str(e)}")
    else:
        st.warning("Please enter a tweet to analyze.")