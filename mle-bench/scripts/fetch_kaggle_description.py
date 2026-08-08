#!/usr/bin/env python3
"""
Fetch Kaggle competition description and convert to markdown.

Usage:
    python fetch_kaggle_description.py <competition-slug>
    python fetch_kaggle_description.py playground-series-s5e8
    
Or with a URL:
    python fetch_kaggle_description.py https://www.kaggle.com/competitions/playground-series-s5e8
"""

import sys
import re
import requests
from bs4 import BeautifulSoup
import html2text


def extract_slug(input_str: str) -> str:
    """Extract competition slug from URL or return as-is if already a slug."""
    # Match Kaggle competition URL patterns
    patterns = [
        r'kaggle\.com/competitions/([^/\s?]+)',
        r'kaggle\.com/c/([^/\s?]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return match.group(1)
    return input_str.strip()


def fetch_competition_description(slug: str) -> str:
    """Fetch the competition overview page and convert to markdown."""
    url = f"https://www.kaggle.com/competitions/{slug}/overview"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    print(f"Fetching: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Kaggle loads content dynamically, so we might need to look for embedded JSON
    # or specific div containers
    
    # Try to find the main content area
    content_divs = soup.find_all('div', class_=re.compile(r'markdown|description|overview'))
    
    if not content_divs:
        # Try getting script tags that might contain the data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'description' in script.string.lower():
                # Try to extract description from JSON
                match = re.search(r'"description"\s*:\s*"([^"]+)"', script.string)
                if match:
                    return match.group(1).encode().decode('unicode_escape')
    
    # Convert HTML to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0  # Don't wrap lines
    
    if content_divs:
        content = '\n'.join(str(div) for div in content_divs)
    else:
        # Fallback to body
        content = str(soup.body) if soup.body else str(soup)
    
    markdown = h.handle(content)
    return markdown


def save_description(slug: str, output_path: str = None):
    """Fetch and save competition description."""
    markdown = fetch_competition_description(slug)
    
    if output_path is None:
        output_path = f"mlebench/competitions/{slug}/description.md"
    
    with open(output_path, 'w') as f:
        f.write(f"# {slug.replace('-', ' ').title()}\n\n")
        f.write(markdown)
    
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_arg = sys.argv[1]
    slug = extract_slug(input_arg)
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Competition slug: {slug}")
    save_description(slug, output_path)
