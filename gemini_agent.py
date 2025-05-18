import os
import time
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import *

load_dotenv()
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class AgentState(TypedDict):
    """Agent state for the graph."""
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


class GEMINI_AGENT:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            max_tokens=1024,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        self.tools = [
            duckduck_websearch,
            serper_websearch,
            visit_webpage,
            wiki_search,
            youtube_viewer,
            text_splitter,
            read_file,
            excel_read,
            csv_read,
            mp3_listen,
            image_caption,
            run_python,
            multiply,
            add,
            subtract,
            divide
        ]

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.app = self._graph_compile()

    def _graph_compile(self):
        builder = StateGraph(AgentState)
        # Define nodes: these do the work
        builder.add_node("assistant", self._assistant)
        builder.add_node("tools", ToolNode(self.tools))
        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        react_graph = builder.compile()
        return react_graph

    def _assistant(self, state: AgentState):
        sys_msg = SystemMessage(
            content=
            """
            You are a helpful assistant tasked with answering questions using a set of tools. When given a question, follow these steps:
            1. Create a clear, step-by-step plan to solve the question.
            2. If a tool is necessary, select the most appropriate tool based on its functionality. If one tool isn't working, use another with similar functionality.
            3. Execute your plan and provide the response in the following format:

            FINAL ANSWER: [YOUR FINAL ANSWER]

            Your final answer should be:

            - A number (without commas or units unless explicitly requested),
            - A short string (avoid articles, abbreviations, and use plain text for digits unless otherwise specified),
            - A comma-separated list (apply the formatting rules above for each element, with exactly one space after each comma).

            Ensure that your answer is concise and follows the task instructions strictly. If the answer is more complex, break it down in a way that follows the format.
            Begin your response with "FINAL ANSWER: " followed by the answer, and nothing else.
            """
        )

        return {
            "messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])],
            "input_file": state["input_file"]
        }

    def extract_after_final_answer(self, text):
        keyword = "FINAL ANSWER: "
        index = text.find(keyword)
        if index != -1:
            return text[index + len(keyword):]
        else:
            return ""

    def run(self, task: dict):
        task_id, question, file_name = task["task_id"], task["question"], task["file_name"]
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        if file_name == "" or file_name is None:
            question_text = question
        else:
            question_text = f'{question} with TASK-ID: {task_id}'
        messages = [HumanMessage(content=question_text)]

        max_retries = 5
        base_sleep = 1
        for attempt in range(max_retries):
            try:
                response = self.app.invoke({"messages": messages, "input_file": None})
                final_ans = self.extract_after_final_answer(response['messages'][-1].content)
                time.sleep(60) # avoid rate limit
                return final_ans
            except Exception as e:
                sleep_time = base_sleep * (attempt + 1)
                if attempt < max_retries - 1:
                    print(str(e))
                    print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return f"Error processing query after {max_retries} attempts: {str(e)}"
        return "This is a default answer."