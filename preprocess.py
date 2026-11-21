"""
Module 2: Text Preprocessing

Cleans raw news article text:
1. Strips dataset noise (e.g. '(CNN) --' or '(CNN) -' prefix headers).
2. Removes leftover HTML tags & unescapes HTML entities.
3. Collapses extra whitespace and newlines.
4. Truncates long articles by word count to fit model token limits.
"""

import re
import html
from typing import Optional


def remove_cnn_prefix(text: str) -> str:
    """Strip out common CNN source noise prefixes."""
    # Matches patterns like "(CNN) --", "(CNN) -", "(CNN) —", "CNN --", etc. at start of text
    pattern = r'^\s*(?:\(CNN\)|CNN)\s*[\-\–\—]*\s*'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)


def remove_html_tags(text: str) -> str:
    """Strip HTML markup and unescape html entities."""
    clean_text = re.sub(r'<.*?>', '', text)
    return html.unescape(clean_text)


def collapse_whitespace(text: str, preserve_paragraphs: bool = True) -> str:
    """Normalize whitespace while optionally preserving natural paragraph breaks."""
    if preserve_paragraphs:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()
    return re.sub(r'\s+', ' ', text).strip()


def truncate_words(text: str, max_words: int = 700) -> str:
    """Truncate article to max word limit if explicitly requested."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text


def preprocess_article(text: str, max_words: Optional[int] = None, preserve_paragraphs: bool = True) -> str:
    """
    Full light preprocessing pipeline for abstractive summarization.
    Preserves whole article content without arbitrary cutoff when max_words is None.

    Args:
        text (str): Raw article text.
        max_words (Optional[int]): Maximum word count limit for truncation. Defaults to None (full article).
        preserve_paragraphs (bool): Whether to keep paragraph breaks. Defaults to True.

    Returns:
        str: Cleaned, natural text ready for model input.
    """
    if not text:
        return ""
    
    cleaned = remove_cnn_prefix(text)
    cleaned = remove_html_tags(cleaned)
    cleaned = collapse_whitespace(cleaned, preserve_paragraphs=preserve_paragraphs)
    if max_words is not None:
        cleaned = truncate_words(cleaned, max_words=max_words)
    
    return cleaned


def main():
    """Standalone CLI test for Module 2."""
    print("--- Testing Module 2: Text Preprocessing ---")
    
    dirty_sample = """
    (CNN) -- <div>The weather today across northern states was <b>unusually severe</b>.</div> 
    
    Emergency services reported  multiple incidents      and warnings were issued nationwide.
    More details were released later in the afternoon...
    """
    
    cleaned_sample = preprocess_article(dirty_sample, max_words=20)
    
    print("\n[Dirty Input]:")
    print(repr(dirty_sample))
    print("\n[Cleaned & Truncated Output]:")
    print(cleaned_sample)
    
    print("\nModule 2 test complete successfully.")


if __name__ == "__main__":
    main()
