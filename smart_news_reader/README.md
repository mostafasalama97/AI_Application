**📌 Project: Smart News Reader**

---------------------------------

**Smart News Reader** is an AI-powered news search application that allows users to search for news articles, summarize them using OpenAI models, and display them in an easy-to-read format.

* * * * *

**🚀 Features**

---------------

-   **🔎 AI-Powered News Search**: Fetches news articles from NewsAPI.

-   **🧠 AI Summarization**: Summarizes news using OpenAI models.

-   **📜 Web Scraping**: Extracts full article text when necessary.

-   **🌍 Multi-Source Search**: Searches across multiple news sources like Google News, BBC, TechCrunch.

-   **📊 React Frontend**: Displays summarized articles in a clean UI.

-   **🌐 Flask API Backend**: Handles AI processing and API requests.

-   **🔗 CORS Support**: Allows frontend-backend communication.

* * * * *

**📂 Project Structure**

------------------------

`smart_news_reader/

│── backend/        # Flask backend with AI-powered news summarization

│── front-end/      # React frontend for user interaction

│── .env            # API keys configuration (not included in repo)`

* * * * *

**⚙️ Setup & Run**

------------------

### **1️⃣ Clone the Repository**

`git clone https://github.com/your-repo/smart_news_reader.git

cd smart_news_reader`

### **2️⃣ Backend Setup (Flask)**

`cd backend

python3 -m venv venv          # Create a virtual environment

source venv/bin/activate       # Activate the virtual environment

pip install -r requirements.txt  # Install dependencies

python main.py                 # Start backend on port 5000`

> **Note:** You need to add your **OpenAI API Key** and **NewsAPI Key** in a `.env` file.

* * * * *

### **3️⃣ Frontend Setup (React)**

`cd ../front-end

npm install                    # Install dependencies

npm start                       # Start React frontend on port 3000`

* * * * *

**📌 Usage**

------------

1️⃣ **Enter your search query** in the input box.

2️⃣ **Select the number of articles** to retrieve.

3️⃣ **Click "Search"** to fetch and summarize news.

4️⃣ **View AI-powered summaries** and **click "Read More"** for full articles.

* * * * *

**🛠️ Tech Stack**

------------------

-   **Backend**: Python, Flask, OpenAI API, NewsAPI, BeautifulSoup

-   **Frontend**: React, Bootstrap, Axios

-   **Database**: No database required (API-based)

* * * * *

**🐛 Troubleshooting**

----------------------

-   **Frontend Error: `Can't resolve 'web-vitals'`**

    `cd front-end

    npm install web-vitals

    npm start`

-   **Backend Missing Dependencies**


    `cd backend

    source venv/bin/activate

    pip install -r requirements.txt`

* * * * *

**📜 License**

--------------

This project is open-source. Feel free to modify and improve it! 🚀