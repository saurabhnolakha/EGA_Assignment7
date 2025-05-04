# Web History Memorizer: Chrome Extension for Smart Web Content Indexing

> Semantic and phrase search for your browsing history with highlighting and navigation.

## Overview

Web History Memorizer is a Chrome extension that automatically indexes web pages you visit, allowing you to later search their content using either semantic (meaning-based) or exact phrase matching. When you find what you're looking for, the extension takes you directly to the relevant page and highlights the content.

## Features

- **Automatic Content Extraction**: Captures page content as you browse
- **Semantic Search**: Find content by meaning, not just keywords
- **Phrase Search**: Find exact text matches with Chrome's native highlighting
- **Content Search**: Find semantic matches with custom highlighting
- **Privacy-Focused**: Locally processed with sensitive domains excluded
- **Fast & Efficient**: Quick search across your browsing history
- **Ollama Integration**: Uses local AI for embeddings

## Quick Start

### Prerequisites

- Python 3.8+
- Chrome browser
- [Ollama](https://ollama.ai/) with the `nomic-embed-text` model installed

### Backend Setup

1. Clone this repository
   ```
   git clone <repository-url>
   cd Assignment7
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Start the API server
   ```
   python api_server.py
   ```

### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked" and select the `chrome_plugin` directory from this repository
4. The Web History Memorizer icon should appear in your Chrome toolbar

### Usage

1. **Indexing Pages**:
   - Simply browse the web! The extension automatically captures and indexes pages you visit.
   - The capture happens in the background and is sent to the local API server.

2. **Searching**:
   - Click the extension icon in the Chrome toolbar
   - Type your search query in the search box
   - Choose either:
     - **Phrase Search**: For exact text matching with Chrome's native highlighting
     - **Content Search**: For semantic/meaning-based search with custom highlighting

3. **Viewing Results**:
   - The search result will open in a new tab
   - For Phrase Search, Chrome will highlight the exact text if available
   - For Content Search, the extension will highlight relevant content

## Architecture

The project consists of two main components:

1. **Backend**: Python-based API server for content extraction, embedding, indexing, and search
2. **Frontend**: Chrome extension for capturing web content and displaying search results

### Backend Components

- **Content Extractor**: Extracts main content from HTML, converts to Markdown
- **Embedding Generator**: Generates embeddings for text chunks using Ollama
- **FAISS Index**: Stores and searches vectors efficiently
- **API Server**: Flask-based server with endpoints for adding content and searching

### Frontend Components

- **Background Script**: Monitors browsing and manages history
- **Content Script**: Extracts page content and handles highlighting
- **Popup UI**: Provides search interface with result display

## Technical Details

### Dependencies

```
flask
flask-cors
faiss-cpu
numpy
requests
beautifulsoup4
markdownify
filelock
```

### API Endpoints

- **POST /add**: Add a new webpage to the index
  ```json
  {
    "url": "https://example.com/page",
    "title": "Example Page",
    "timestamp": 1647252413000
  }
  ```

- **POST /search**: Search the index
  ```json
  {
    "query": "example search term",
    "k": 5,
    "model": "ollama",
    "rerank": true
  }
  ```

### Chunking Strategy

Content is broken into 50-word chunks with 15-word overlaps for optimal search precision and highlighting accuracy.

### FAISS Index

Uses a flat L2 index for maximum recall, with metadata stored separately to map vectors back to source content.

### Chrome Extension

- **Permissions**: tabs, scripting, storage, webNavigation, alarms
- **Background Script**: Handles periodic history capture and API communication
- **Content Script**: Manages content extraction and highlighting
- **Text Fragment Support**: Uses Chrome's `#:~:text=` feature for native highlighting

## Development

### Project Structure

```
Assignment7/
├── api_server.py               # Flask API server
├── embedding_generator.py      # Gemini embedding generator (optional)
├── embedding_generator_ollama.py # Ollama embedding generator
├── content_extractor.py        # HTML to Markdown extraction
├── faiss_index_ollama.bin      # Ollama FAISS index (binary)
├── metadata_ollama.json        # Ollama metadata for chunks
├── requirements.txt            # Python dependencies
├── chrome_plugin/              # Chrome extension files
│   ├── manifest.json           # Extension manifest
│   ├── background.js           # Background script
│   ├── content.js              # Content script
│   ├── popup.html              # Popup UI HTML
│   └── popup.js                # Popup UI logic
└── extracted_content/          # Extracted Markdown files
```

### Setup for Development

1. Follow the Quick Start instructions
2. For development:
   - Chrome extension changes: Make changes and click "Reload" in `chrome://extensions/`
   - Backend changes: Restart the API server

### Future Enhancements

- **CLI Tool**: For testing and managing the index
- **End-to-End Testing**: For validating the entire pipeline
- **Scroll-to-Highlight**: Automatically scroll to relevant content
- **Clear Highlight**: Button to remove highlights
- **Multi-Model Support**: Add more embedding models

## License

This project is open source and available under the MIT License. 