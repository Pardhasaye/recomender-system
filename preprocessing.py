
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Contractions dictionary
contractions = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "cause": "because", "could've": "could have", "couldn't": "could not", 
    "didn't": "did not", "doesn't": "does not", "don't": "do not"
}

def clean_text(text, remove_stopwords=True):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == "": return ""
    
    text = str(text).lower()
    # Sort contractions to handle longer matches first (e.g., "can't've" before "can't")
    sorted_contractions = sorted(contractions.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join([re.escape(k) for k in sorted_contractions]) + r')\b', flags=re.IGNORECASE)
    text = pattern.sub(lambda m: contractions[m.group(0).lower()], text)
    
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[_"\\\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" 
        u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    words = word_tokenize(text)
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if w not in stop_words and len(w) > 2]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)
