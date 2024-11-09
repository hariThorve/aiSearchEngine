





# 🔍 AI Research Assistant

An intelligent search engine that combines web scraping, vector databases, and Google's Gemini AI to provide comprehensive answers to queries by analyzing multiple sources and synthesizing information.

## 🌟 Features

- **Smart Search**: Utilizes DuckDuckGo API with fallback mechanisms
- **AI-Powered Summaries**: Generates concise, relevant summaries using Google's Gemini AI
- **Vector Database**: Stores and retrieves information efficiently using ChromaDB
- **Programming Focus**: Special handling for programming-related queries
- **Source Verification**: Provides original sources with content previews
- **Progress Tracking**: Real-time progress indicators for search operations
- **Cache Management**: Efficient caching system for faster responses

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```plaintext
GEMINI_API_KEY=your_gemini_api_key_here
```

## 📦 Requirements

```plaintext
streamlit
chromadb
sentence-transformers
duckduckgo-search
requests
beautifulsoup4
google-generativeai
python-dotenv
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter your search query in the input box

4. Adjust the number of sources using the sidebar slider

5. View AI-generated summaries and source details

## 🔧 Configuration

- **Number of Sources**: Adjust via sidebar slider (1-10)
- **Cache Management**: Clear cache using the sidebar button
- **Safety Settings**: Configurable content filtering
- **Search Parameters**: Customizable in `search_web` function

## 🎯 Features in Detail

### Search Capabilities
- Web search with fallback mechanisms
- Programming-specific search optimization
- Source relevance scoring
- Duplicate removal

### AI Integration
- Google Gemini AI integration
- Context-aware summarization
- Programming query detection
- Safety filters

### Content Processing
- Smart content extraction
- HTML cleaning
- Text formatting
- Progress tracking

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Google Gemini AI for providing the language model
- ChromaDB for vector database functionality
- DuckDuckGo for search API
- Streamlit for the web interface

