#!/usr/bin/env python3
"""
Content Extractor Module

This module extracts readable content from HTML and converts it to Markdown format,
filtering out specific domains and removing unnecessary elements.
"""

import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md


# List of domains to exclude from processing
EXCLUDED_DOMAINS = [
    'gmail.com',
    'mail.google.com',
    'youtube.com',
    'github.com',
    'facebook.com',
    'you.com',
    'whatsapp.com',
    'web.whatsapp.com',
    'drive.google.com',
    'docs.google.com'
]

# CSS selectors for common noisy elements to remove
NOISE_SELECTORS = [
    'script', 'style', 'noscript',
    'iframe', 'svg', 'canvas',
    '.ad', '.ads', '.advertisement',
    '.banner', '.cookie-banner',
    '.popup', '.modal',
    '.newsletter', '.subscribe',
    '.comments', '.comment-section',
    '.social-buttons', '.social-share',
    '.related-content'
]

# CSS selectors for potential main content containers
CONTENT_SELECTORS = [
    'main', 'article', '[role="main"]',
    '#bodyContent', '#mw-content-text', '#content', '#main', 
    '.content', '.main', '.article', '.post',
    '[id*="content"]', '[id*="article"]', '[class*="content"]', '[class*="article"]'
]

# CSS selectors for navigational elements
NAV_SELECTORS = [
    'header', 'footer', 'nav', 'aside',
    '#header', '#footer', '#navigation', '#sidebar', '#menu',
    '.header', '.footer', '.navigation', '.sidebar', '.menu', '.nav',
    '[role="navigation"]', '[role="banner"]', '[role="complementary"]'
]


def is_allowed_domain(url):
    """
    Check if the URL is from an allowed domain (not in excluded list).
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the domain is allowed, False otherwise
    """
    if not url:
        return False
        
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if the domain or any parent domain is in the excluded list
        for excluded in EXCLUDED_DOMAINS:
            if domain == excluded or domain.endswith(f'.{excluded}'):
                return False
        return True
    except:
        return False


def clean_html(html_content):
    """
    Remove ads, navigation, and other noise from HTML.
    
    Args:
        html_content (str): The raw HTML content
        
    Returns:
        BeautifulSoup: A cleaned BeautifulSoup object
    """
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # First pass: Remove script, style, and other definitely unwanted elements
    for selector in NOISE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()
    
    # Return the soup for further processing
    return soup


def extract_main_content(soup):
    """
    Extract the main content from a BeautifulSoup object.
    
    Args:
        soup (BeautifulSoup): The soup object to process
        
    Returns:
        BeautifulSoup or element: The main content element or the original soup
    """
    # Try to find the main content container
    # 1. First try explicit content selectors
    for selector in CONTENT_SELECTORS:
        content_elements = soup.select(selector)
        if content_elements:
            # If multiple elements match, use the one with the most text
            if len(content_elements) > 1:
                content_elements.sort(key=lambda x: len(x.get_text()), reverse=True)
            
            # Return the largest content element
            return content_elements[0]
    
    # 2. If no content selectors matched, remove navigation and try again
    for selector in NAV_SELECTORS:
        for element in soup.select(selector):
            element.decompose()
    
    # 3. Find the element with the most text
    def get_element_text_length(element):
        if element.name in ['script', 'style', 'header', 'footer', 'nav']:
            return 0
        return len(element.get_text())
    
    all_elements = soup.find_all(['div', 'section', 'main', 'article'])
    if all_elements:
        all_elements.sort(key=get_element_text_length, reverse=True)
        if get_element_text_length(all_elements[0]) > 200:  # Minimum text threshold
            return all_elements[0]
    
    # If all else fails, return the body or the original soup
    body = soup.find('body')
    return body if body else soup


def html_to_markdown(html_content):
    """
    Convert cleaned HTML to Markdown.
    
    Args:
        html_content (str or BeautifulSoup): The HTML content to convert
        
    Returns:
        str: Markdown text
    """
    # Configure markdownify to ignore links and images
    markdown_text = md(str(html_content), strip=['a', 'img'], heading_style='atx')
    
    # Clean up the markdown
    # Remove multiple blank lines
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    # Trim leading/trailing whitespace
    markdown_text = markdown_text.strip()
    
    return markdown_text


def extract_content(html_content, url):
    """
    Main function to extract and convert content from HTML to Markdown.
    
    Args:
        html_content (str): The raw HTML content
        url (str): The source URL
        
    Returns:
        str or None: Markdown text if URL is allowed, None otherwise
    """
    # Check if the domain is allowed
    if not is_allowed_domain(url):
        return None
    
    # Clean the HTML
    cleaned_soup = clean_html(html_content)
    
    # Extract main content
    main_content = extract_main_content(cleaned_soup)
    
    # Convert to markdown
    markdown_content = html_to_markdown(main_content)
    
    # Return None if no significant content was found
    if len(markdown_content) < 100:
        return None
        
    return markdown_content


# Utility function for testing
def extract_content_from_file(html_file, url):
    """
    Extract content from an HTML file.
    
    Args:
        html_file (str): Path to HTML file
        url (str): URL associated with the HTML content
        
    Returns:
        str or None: Markdown text if URL is allowed, None otherwise
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return extract_content(html_content, url)


if __name__ == "__main__":
    # Simple test code when module is run directly
    import sys
    
    if len(sys.argv) > 1:
        test_html = sys.argv[1]
        test_url = sys.argv[2] if len(sys.argv) > 2 else "http://example.com"
        
        result = extract_content_from_file(test_html, test_url)
        if result:
            print(result)
        else:
            print(f"The URL {test_url} is not allowed or couldn't be processed.")
    else:
        print("Usage: python content_extractor.py <html_file> [url]") 