# Submission to 20 Questions in GAIA Benchmark
GAIA benchmark is a benchmark to evaluate the ability of agent to call tools, feel free to dive into [the details](https://openreview.net/forum?id=fibxvahvs3).

Agent Course provided by HuggingFace only selects 20 questions as its final assessment. The goal to pass this assessment is to complete at least 6 of them correctly.

## <b>Details about the Submission</b>

This agent in [gemini_agent.py](./gemini_agent.py) is build based on the framework of LangChain and its core LLM model is GEMINI-2.0 with up to 16 tools (eg. web_search, wiki_search, read_file, mp3_listen, ... ... )

To run this agent, you need to pass several API keys (they are <b>free</b>) in [.env](./env): 
- <b>GEMINI_API</b>: To access GEMINI Model
- <b>SERPER_API</b>: To support web search tools
- <b>ASSMEBLY_API</b>: For the tool to listen to mp3

Finally, this agent could complete up to six tasks. It still has the room for improvement. 

Feel free to clone it to HuggingFace hub and run it. 