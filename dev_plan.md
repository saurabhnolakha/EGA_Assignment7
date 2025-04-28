# Chrome Plugin Web Page Indexing Plan

## Overall Architecture
1. **Backend Component**: Python scripts to handle embeddings, FAISS indexing, and searching.
2. **Frontend Component**: Chrome plugin to capture webpage content and handle search UI.
3. **Integration Layer**: API or file-based communication between plugin and backend.

## Step-by-Step Implementation Plan

### Phase 1: Backend Development

1. **Module 1: Web Content Extractor** (`content_extractor.py`) ✅
   - Create a function to extract readable text from HTML content.
   - Handle text cleaning, removing ads, navigation menus, etc.
   - Convert content to Markdown format
   - Filter out sensitive domains (Gmail, WhatsApp, etc.)

2. **Module 2: Embedding Generator** (`embedding_generator.py`) 🔄
   - Set up embedding model (Nomic or alternative local model).
   - Create function to generate embeddings for text chunks.
   - Implement chunking system with overlap
   - Add caching to avoid redundant API calls

3. **Module 3: FAISS Index Builder** (`index_builder.py`)
   - Create functions to build and update FAISS index from embeddings.
   - Add serialization/deserialization to save and load index files.
   - Include metadata storage for mapping chunks to URLs

4. **Module 4: Search Engine** (`search_engine.py`)
   - Build search function using FAISS for semantic similarity.
   - Include metadata retrieval and result ranking.
   - Support for various search options (top-k results, etc.)

5. **Module 5: API Server** (`api_server.py`)
   - Simple Flask API to expose backend functionality to Chrome plugin.
   - Endpoints for adding page content and performing searches.
   - Support for bulk operations

### Phase 2: Integration & Testing

6. **Module 6: Command Line Tool** (`cli.py`)
   - Build CLI to test and manage index without plugin.
   - Include commands for manual indexing of URLs and testing searches.
   - Support for index management

7. **Module 7: End-to-End Testing** (`e2e_tests.py`)
   - Create script to verify full pipeline using sample pages.
   - Validate search result quality and highlighting functionality.
   - Performance testing and optimization

### Phase 3: Chrome Plugin Development

8. **Plugin Structure Setup**
   - Manifest.json configuration
   - Content scripts for page interaction
   - Background script for API communication

9. **Content Capture System**
   - JavaScript to extract page content on visit
   - Skip functionality for confidential pages
   - Queue system for processing

10. **Search UI**
    - Simple popup interface for search queries
    - Results display with links to pages
    - Highlighting functionality

11. **Text Highlighting**
    - System to locate and highlight search matches on target pages
    - Scroll to relevant content

## Implementation Strategy

1. Develop and test each backend module independently with simple test cases.
2. Combine backend modules and test with realistic webpage samples.
3. Set up basic API server to expose functionality.
4. Build minimal Chrome plugin that communicates with backend.
5. Incrementally enhance plugin features and UI.

## Test Directory Structure
All test scripts, sample data, and test outputs will be stored in `Assignment7/test/`

## Current Progress

- ✅ Module 1: Web Content Extractor
  - Successfully extracts main content from HTML
  - Converts to Markdown format
  - Filters sensitive domains
  - Handles a variety of websites without site-specific rules

- 🔄 Module 2: Embedding Generator (In Progress)
  - Planning completed
  - Implementation pending

- ⏱️ Remaining modules planned but not started 