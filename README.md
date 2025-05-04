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

#### Content Extractor (`content_extractor.py`)
- **Purpose**: Extracts relevant content from HTML pages while filtering out ads, navigation menus, and other noise
- **Implementation**: Uses BeautifulSoup4 to parse HTML and customized algorithms to identify main content
- **Features**:
  - Converts HTML to clean Markdown format for better text processing
  - Filters out sensitive domains (email, banking, etc.)
  - Preserves important structural elements (headings, lists, links)
  - Domain-agnostic approach works across various website layouts

#### Embedding Generator (`embedding_generator_ollama.py`)
- **Purpose**: Converts text chunks into numerical vector representations for semantic search
- **Implementation**: Communicates with the Ollama API to generate embeddings using the `nomic-embed-text` model
- **Features**:
  - Breaks content into 50-word chunks with 15-word overlap
  - Manages embedding generation with error handling
  - Caches embeddings to avoid redundant processing
  - Handles rate limiting and retries

#### FAISS Index Management
- **Purpose**: Efficiently stores and retrieves vector embeddings
- **Implementation**: Uses Facebook AI Similarity Search (FAISS) library for vector storage and similarity search
- **Features**:
  - Uses a flat L2 index for maximum recall
  - Serializes index to disk for persistence (`faiss_index_ollama.bin`)
  - Supports incremental updates as new content is indexed
  - Optimized for fast similarity searches

#### API Server (`api_server.py`)
- **Purpose**: Provides HTTP endpoints for the Chrome extension to interact with the backend
- **Implementation**: Flask-based web server with JSON API endpoints
- **Features**:
  - `/add` endpoint for adding new pages to the index
  - `/search` endpoint for querying the index
  - CORS support for browser security
  - Mutex locking to prevent race conditions
  - Duplicate URL detection to avoid redundant processing

### Frontend Components

#### Background Script (`background.js`)
- **Purpose**: Monitors browsing activity and manages the extension's background processes
- **Implementation**: Chrome extension service worker responding to browser events
- **Features**:
  - Listens for tab navigation events to capture page visits
  - Manages periodic catch-up for pages that might have been missed
  - Maintains browsing history in local storage
  - Handles communication between popup and content scripts

#### Content Script (`content.js`)
- **Purpose**: Interacts with web pages for content extraction and highlighting
- **Implementation**: JavaScript injected into web pages
- **Features**:
  - Sends page metadata (URL, title, timestamp) to backend for indexing
  - Implements custom highlighting for Content Search
  - Traverses the DOM to find and highlight text matches
  - Uses mutation observers for dynamic content handling

#### Popup UI (`popup.html`, `popup.js`)
- **Purpose**: Provides user interface for search functionality
- **Implementation**: HTML/JavaScript popup activated when clicking the extension icon
- **Features**:
  - Search input field and execution buttons
  - Two search modes (Phrase Search and Content Search)
  - Displays status messages and handles errors gracefully
  - Communicates with both the background script and backend API

### Data Flow

1. **Content Capture Flow**:
   - User visits a webpage → Content script captures URL, title, and timestamp
   - Data sent to `/add` endpoint → Content extraction → Chunking → Embedding → Index update

2. **Search Flow**:
   - User enters query in popup → Query sent to `/search` endpoint
   - Backend generates query embedding → FAISS search → Results ranked (with optional reranking)
   - Results returned to popup → Selected result opened in new tab → Text highlighted

3. **Highlighting Flow**:
   - **Phrase Search**: Uses Chrome's native `#:~:text=` URL fragment
   - **Content Search**: Injects content script → Traverses DOM → Wraps matching text in highlight elements

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