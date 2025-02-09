import os
from typing import TypedDict, Annotated, List
from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod
from datetime import datetime
import re

from getpass import getpass
from dotenv import load_dotenv

from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup

# from IPython.display import display, Image as IPImage
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS


# check if .env exist
if os.path.exists(".env"):
    load_dotenv()
else:
    # ask for API Keys
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")
    os.environ["NEWSAPI_KEY"] = getpass("Enter your News API key: ")

# sets the openAI model to use and initialize model
model = "gpt-4o-mini"
llm= ChatOpenAI(temperature=0, model=model)

# initialize newsapi client
newsapi_key = os.getenv("NEWSAPI_KEY")
if newsapi_key:
    print("newsapi key found successfully")


# for test open ai key
# llm.invoke("Why is the sky blue?").content

# Initialize Flask app
flask_app = Flask(__name__)
CORS(flask_app)  # Allow React frontend to access API


newsapi = NewsApiClient(api_key=newsapi_key)
# query = "ai news of the day" # replace with your search query
# all_articles = newsapi.get_everything(
#     q=query,
#     sources='google-news,bbc-news,techcrunch',
#     domains='bbc.co.uk,techcrunch.com',
#     language='en',
#     sort_by='relevancy',
# )

# all_articles = all_articles['articles']

# # print all articles
# for article in all_articles:
#     print(article['title'])
#     print(article['url'])
#     print(article['description'])
#     print("")

# define data structure
class GraphState(TypedDict):
    news_query: Annotated[str, "User's search query"]
    num_searches_remaining: Annotated[int, "Remaining searches"]
    newsapi_params: Annotated[dict, "Parameters for the News API"]
    past_searches: Annotated[List[dict], "Previous search terms"]
    articles_metadata: Annotated[list[dict], "Metadata of retrieved articles"]
    scraped_urls: Annotated[List[str], "List of scraped URLs"]
    num_articles_tldr: Annotated[int, "Number of articles to summarize"]
    potential_articles: Annotated[List[dict[str, str, str]], "Full text of selected articles"]
    tldr_articles: Annotated[List[dict[str, str, str]], "Summarized articles"]
    formatted_results: Annotated[str, "Final output"]

# Define NewsAPI argument data structure with Pydantic

class NewsApiParams(BaseModel):
    q: str = Field(description="Keyword search terms")
    sources: str = Field(description="Comma-separated news sources")
    from_param: str = Field(description="Start date (YYYY-MM-DD)")
    to: str = Field(description="End date (YYYY-MM-DD)")
    language: str = Field(description="Article language")
    sort_by: str = Field(description="Sorting method")


def generate_newsapi_params(state: GraphState) -> GraphState:
    """Based on the query, generate News API params."""
    # Initialize parser to define the structure of the response
    parser = JsonOutputParser(pydantic_object=NewsApiParams)
    
    # Retrieve today's date
    today = datetime.now().strftime("%Y-%m-%d")

    # Retrieve list of past search params
    past_searches = state.get("past_searches", [])

    # Retrieve number of searches remaining
    num_searches_remaining = state["num_searches_remaining"]

    # Retrieve user query
    news_query = state["news_query"]

    # Fix: Use .get_format_instructions() instead of trying to access it as an attribute
    format_instructions = parser.get_format_instructions()

    template = """
    Today is {today_date}.

    Create a param dict for the News API based on the user query:
    {query}

    These searches have already been made. Loosen the search terms to get more results.
    {past_searches}
    
    Following these formatting instructions:
    {format_instructions}

    Including this one, you have {num_searches_remaining} searches remaining.
    If this is your last search, use all news sources and a 30 days search range.
    """

    # Create a prompt template to merge the query, today's date, and the format instructions
    prompt_template = PromptTemplate(
        template=template,
        variables={
            "query": news_query, 
            "today_date": today, 
            "format_instructions": format_instructions,  # Fix applied here
            "past_searches": past_searches, 
            "num_searches_remaining": num_searches_remaining
        },
        partial_variables={"format_instructions": format_instructions}
    )

    # Create prompt chain template
    chain = prompt_template | llm | parser

    # Invoke the chain with the news API query
    result = chain.invoke({
        "query": news_query, 
        "today_date": today, 
        "past_searches": past_searches, 
        "num_searches_remaining": num_searches_remaining
    })

    # Fix: Assign the parsed result correctly
    state["newsapi_params"] = result

    return state



def retrieve_articles_metadata(state: GraphState) -> GraphState:
    """Using the NewsAPI params, perform api call."""
    # parameters generated for the News API
    newsapi_params = state.get("newsapi_params", {})


    # decrement the number of searches remaining
    state["num_searches_remaining"] -= 1

    try:
        # create a NewsApiClient object
        newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))
        
        # retreive the metadata of the new articles
        articles = newsapi.get_everything(**newsapi_params)

        # append this search term to the past searches to avoid duplicates
        state['past_searches'].append(newsapi_params)

        # load urls that have already been returned and scraped
        scraped_urls = state["scraped_urls"]

        # filter out articles that have already been scraped
        new_articles = []
        for article in articles['articles']:
            if article['url'] not in scraped_urls and len(state['potential_articles']) + len(new_articles) < 10:
                new_articles.append(article)

        # reassign new articles to the state
        state["articles_metadata"] = new_articles

    # handle exceptions
    except Exception as e:
        print(f"Error: {e}")

    return state



def retrieve_articles_text(state: GraphState) -> GraphState:
    """Web scrapes to retrieve article text."""
    # load retrieved article metadata
    article_metadata= state["articles_metadata"]
    # Add headers to simulate a browser
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # create list to store valid article dicts
    potential_articles= []
    # iterate over the urls
    for article in article_metadata:
        # extract the url
        url= article['url']

        # use beautiful soup to extract the article content
        response= requests.get(url, headers=headers)
        # check if the request was successful
        if response.status_code == 200:
            # parse the HTML content
            soup= BeautifulSoup(response.content, 'html.parser')
            # find the article content
            text=soup.get_text(strip=True)
            # append article dict to list
            potential_articles.append({"title": article["title"], "url": url, "description": article["description"], "text": text})
            # append the url to the processed urls
            state["scraped_urls"].append(url)
    # append the processed articles to the state
    state["potential_articles"].extend(potential_articles)

    return state

def select_top_urls(state: GraphState) -> GraphState:
    """Based on the article synoses, choose the top-n articles to summarize."""
    news_query=state["news_query"]
    num_articles_tldr=state["num_articles_tldr"]
    # load all processed articles with full text but no summaries
    potential_articles= state["potential_articles"]
    # format the metadata
    formatted_metadata= "\n".join([f"{article['title']} - {article['url']}" for article in potential_articles])
    prompt = f"""
    Based on the user news query:
    {news_query}

    Reply with a list of strings of up to {num_articles_tldr} relevant urls.
    Don't add any urls that are not relevant or aren't listed specifically.
    {formatted_metadata}
    """
    result = llm.invoke(prompt).content
    # use regex to extract the urls as a list
    url_pattern = r'(https?://[^\s",]+)'
    # Find all URLs in the text
    urls = re.findall(url_pattern, result)
    # add the selected article metadata to the state
    tldr_articles= [article for article in potential_articles if article['url'] in urls]
    # tldr_articles = [article for article in potential_articles if article['url'] in urls]
    state["tldr_articles"] = tldr_articles

    return state

async def summarize_articles_parallel(state: GraphState) -> GraphState:
    """Summarize the articles based on full text."""
    tldr_articles = state["tldr_articles"]

    # prompt = """
    # Summarize the article text in a bulleted tl;dr. Each line should start with a hyphen -
    # {article_text}
    # """

    prompt = """
    Create a * bulleted summarizing tldr for the article:
    {text}
    
    Be sure to follow the following format exaxtly with nothing else:
    {title}
    {url}
    * tl;dr bulleted summary
    * use bullet points for each sentence
    """

    # iterate over the selected articles and collect summaries synchronously
    for i in range(len(tldr_articles)):
        text = tldr_articles[i]["text"]
        title = tldr_articles[i]["title"]
        url = tldr_articles[i]["url"]
        # invoke the llm synchronously
        result = llm.invoke(prompt.format(title=title, url=url, text=text))
        tldr_articles[i]["summary"] = result.content

    state["tldr_articles"] = tldr_articles

    return state


def format_results(state: GraphState) -> GraphState:
    """Format the results for display."""
    # load a list of past search queries
    q = [newsapi_params["q"] for newsapi_params in state["past_searches"]]
    formatted_results = f"Here are the top {len(state['tldr_articles'])} articles based on search terms:\n{', '.join(q)}\n\n"

    # load the summarized articles
    tldr_articles = state["tldr_articles"]

    # format article tl;dr summaries
    tldr_articles = "\n\n".join([f"{article['summary']}" for article in tldr_articles])

    # concatenate summaries to the formatted results
    formatted_results += tldr_articles

    state["formatted_results"] = formatted_results

    return state


def articles_text_decision(state: GraphState) -> str:
    """Check results of retrieve_articles_text to determine next step."""
    if state["num_searches_remaining"] == 0:
        # if no articles with text were found return END
        if len(state["potential_articles"]) == 0:
            state["formatted_results"] = "No articles with text found."
            return "END"
        # if some articles were found, move on to selecting the top urls
        else:
            return "select_top_urls"
    else:
        # if the number of articles found is less than the number of articles to summarize, continue searching
        if len(state["potential_articles"]) < state["num_articles_tldr"]:
            return "generate_newsapi_params"
        # otherwise move on to selecting the top urls
        else:
            return "select_top_urls"
        

workflow = Graph()

workflow.set_entry_point("generate_newsapi_params")

workflow.add_node("generate_newsapi_params", generate_newsapi_params)
workflow.add_node("retrieve_articles_metadata", retrieve_articles_metadata)
workflow.add_node("retrieve_articles_text", retrieve_articles_text)
workflow.add_node("select_top_urls", select_top_urls)
workflow.add_node("summarize_articles_parallel", summarize_articles_parallel)
workflow.add_node("format_results", format_results)
# workflow.add_node("add_commentary", add_commentary)

workflow.add_edge("generate_newsapi_params", "retrieve_articles_metadata")
workflow.add_edge("retrieve_articles_metadata", "retrieve_articles_text")
# # if the number of articles with parseable text is less than number requested, then search for more articles
workflow.add_conditional_edges(
    "retrieve_articles_text",
    articles_text_decision,
    {
        "generate_newsapi_params": "generate_newsapi_params",
        "select_top_urls": "select_top_urls",
        "END": END
    }
    )
workflow.add_edge("select_top_urls", "summarize_articles_parallel")
workflow.add_conditional_edges(
    "summarize_articles_parallel",
    lambda state: "format_results" if len(state["tldr_articles"]) > 0 else "END",
    {
        "format_results": "format_results",
        "END": END
    }
    )
workflow.add_edge("format_results", END)

app = workflow.compile()

# Display Graph Structure

# display(
#     IPImage(
#         app.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )


# Run Workflow Function
async def run_workflow(query: str, num_searches_remaining: int = 10, num_articles_tldr: int = 3):
    """Runs LangGraph workflow and returns summarized results."""
    initial_state = {
        "news_query": query,
        "num_searches_remaining": num_searches_remaining,
        "newsapi_params": {},
        "past_searches": [],
        "articles_metadata": [],
        "scraped_urls": [],
        "num_articles_tldr": num_articles_tldr,
        "potential_articles": [],
        "tldr_articles": [],
        "formatted_results": "No articles with text found."
    }
    try:
        result = await app.ainvoke(initial_state)
        return {"articles": result["formatted_results"]}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}


@flask_app.route("/search", methods=["POST"])
def search_news():
    """Handles incoming search requests."""
    data = request.json
    query = data.get("query", "")
    num_articles_tldr = int(data.get("num_articles_tldr", 3))

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400


    # Fetch news from NewsAPI
    newsapi_params = {
        "q": query,
        "sources": "google-news,bbc-news,techcrunch",
        "domains": "bbc.co.uk,techcrunch.com",
        "language": "en",
        "sort_by": "relevancy",
    }
    
    try:
        articles = newsapi.get_everything(**newsapi_params)["articles"]
        formatted_articles = [{"title": a["title"], "url": a["url"], "description": a["description"]} for a in articles]

        # Run AI summarization workflow
        result = asyncio.run(run_workflow(query, num_articles_tldr))

        # Combine original articles + AI summaries
        return jsonify({
            "original_articles": formatted_articles,
            "summarized_articles": result["articles"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start Flask app
if __name__ == "__main__":
    flask_app.run(port=8009, debug=True)