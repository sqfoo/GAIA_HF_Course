import os
import re
import requests
import pandas as pd
from typing import List
from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_core.tools import tool
from langchain.document_loaders import WebBaseLoader
from langchain_experimental.tools import PythonREPLTool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import ImageCaptionLoader, AssemblyAIAudioTranscriptLoader


load_dotenv()
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def duckduck_websearch(query: str) -> str:
    """
    Performs a web search using the given query, downloads the content of two relevant web pages,
    and returns their combined content as a raw string.

    This is useful when the task requires analysis of web page content, such as retrieving poems, 
    changelogs, or other textual resources.

    Args:
        query (str): The search query.

    Returns:
        str: The combined raw text content of the two retrieved web pages.
    """
    search_engine = DuckDuckGoSearchResults(output_format="list", num_results=2)
    page_urls = [url["link"] for url in search_engine(query)]

    loader = WebBaseLoader(web_paths=(page_urls))
    docs = loader.load()

    combined_text = "\n\n".join(doc.page_content[:15000] for doc in docs)

    # Clean up excessive newlines, spaces and strip leading/trailing whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', combined_text).strip()
    cleaned_text = re.sub(r'[ \t]{6,}', ' ', cleaned_text)

    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def serper_websearch(query: str) -> str:
    """
    Performs a web search using the given query with SERPER Search Engine

    Args:
        query (str): The search query.
    
    Returns:
        str: the search result
    """
    search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
    results = search.run(query)
    return results

def visit_webpage(url: str) -> str:
    """
    Fetches raw HTML content of a web page.
    
    Args:
        url: the webpage url
    
    Returns:
        str: The combined raw text content of the webpage
    """
    try:
        response = requests.get(url, timeout=5)
        return response.text[:5000]
    except Exception as e:
        return f"[ERROR fetching {url}]: {str(e)}"

def wiki_search(query: str) -> str:
    """
    Searches for a Wikipedia articles using the provided query and returns the content of the corresponding Wikipedia pages.

    Args:
        query (str): The search term to look up on Wikipedia.

    Returns:
        str: The text content of the Wikipedia articles related to the query.
    """
    retriever = WikipediaRetriever()
    docs = retriever.invoke(query)
    combined_text = "\n\n".join(doc.page_content for doc in docs)
    return combined_text

def youtube_viewer(youtube_url: str, question: str) -> str:
    """
    Analyzes a YouTube video from the provided URL and returns an answer 
    to the given question based on the analysis results.

    Args:
        youtube_url (str): The URL of the YouTube video, in the format 
            "https://www.youtube.com/...".
        question (str): A question related to the content of the video.

    Returns:
        str: An answer to the question based on the video's content.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model='models/gemini-2.5-flash-preview-04-17',
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=youtube_url)
                ),
                types.Part(text=question)
            ]
        )
    )
    return response.text

def text_splitter(text: str) -> List[str]:
    """
    Splits text into chunks using LangChain's CharacterTextSplitter.
    
    Args:
        text: A string of text to split.
    
    Returns:
        List[str]: a list of split text
    """
    splitter = CharacterTextSplitter(chunk_size=450, chunk_overlap=10)
    return splitter.split_text(text)

def read_file(task_id: str) -> str:
    """
    First download the file, then read its content
    
    Args:
        dir: the task_id
    
    Returns:
        str: the file content
    """
    file_url = f'{DEFAULT_API_URL}/files/{task_id}'
    r = requests.get(file_url, timeout=15, allow_redirects=True)
    with open('temp', "wb") as fp:
        fp.write(r.content)
    with open('temp') as f:
        return f.read()

def excel_read(task_id: str) -> str:
    """
    First download the excel file, then read its content
    
    Args:
        dir: the task_id
    
    Returns:
        str: the content of excel file
    """
    try:
        file_url = f'{DEFAULT_API_URL}/files/{task_id}'
        r = requests.get(file_url, timeout=15, allow_redirects=True)
        with open('temp.xlsx', "wb") as fp:
            fp.write(r.content)
        # Read the Excel file
        df = pd.read_excel('temp.xlsx')
        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        return result
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"
   
def csv_read(task_id: str) -> str:
    """
    First download the csv file, then read its content
    
    Args:
        dir: the task_id
    
    Returns:
        str: the content of csv file
    """
    try:
        file_url = f'{DEFAULT_API_URL}/files/{task_id}'
        r = requests.get(file_url, timeout=15, allow_redirects=True)
        with open('temp.csv', "wb") as fp:
            fp.write(r.content)
        # Read the CSV file
        df = pd.read_csv('temp.csv')
        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        return result
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"


def mp3_listen(task_id: str) -> str:
    """
    First download the mp3 file, then listen to it
    
    Args:
        dir: the task_id
    
    Returns:
        str: the content of mp3 file
    """
    file_url = f'{DEFAULT_API_URL}/files/{task_id}'
    r = requests.get(file_url, timeout=15, allow_redirects=True)
    with open('temp.mp3', "wb") as fp:
        fp.write(r.content)
    loader = AssemblyAIAudioTranscriptLoader(file_path="temp.mp3", api_key=os.getenv("AssemblyAI_API_KEY"))
    docs = loader.load()
    contents = [doc.page_content for doc in docs]
    return "\n".join(contents)
    

def image_caption(dir: str) -> str:
    """
    Understand the content of the provided image
    
    Args:
        dir: the image url link
    
    Returns:
        str: the image caption
    """
    loader = ImageCaptionLoader(images=[dir])
    metadata = loader.load()
    return metadata[0].page_content


def run_python(code: str):
    """ Run the given python code
    
    Args:
        code: the python code
    """
    return PythonREPLTool().run(code)

def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    
    Args:
        a: first float
        b: second float
    
    Returns:
        float: the multiplication of a and b
    """
    return a * b

def add(a: float, b: float) -> float:
    """
    Add two numbers.
    
    Args:
        a: first float
        b: second float
    
    Returns:
        float: the sum of a and b
    """
    return a + b

def subtract(a: float, b: float) -> float:
    """
    Subtract two numbers.
    
    Args:
        a: first float
        b: second float
    
    Returns:
        float: the result after a subtracted by b
    """
    return a - b

def divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: first float
        b: second float
    
    Returns:
        float: the result after a divided by b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b