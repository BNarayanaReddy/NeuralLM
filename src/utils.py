import re
import unicodedata

def normalize_text(text, lowercase = True):
    if lowercase:
        text = text.lower()
    # Unicode Normalization (e.g., converting fancy quotes to standard ones)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\*_`\[\]\^{}]', '', text) # handle _word_ and *word*
    text = re.sub(r'\b(mr|mrs|ms|dr|st)\.', r'\1', text, flags=re.IGNORECASE) # handle mr. mrs. etc
    text = text.strip()
    return text