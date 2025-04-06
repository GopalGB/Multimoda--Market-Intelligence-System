# data/preprocessing/text_preprocessor.py
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import torch

class TextPreprocessor:
    """
    Text preprocessor for media content analysis.
    Handles cleaning, normalization, tokenization, and feature extraction.
    """
    def __init__(
        self,
        language: str = "en",
        use_spacy: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Language code
            use_spacy: Whether to use spaCy for processing
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
        """
        self.language = language
        self.use_spacy = use_spacy
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize spaCy if needed
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load(f"{language}_core_web_sm")
            except OSError:
                print(f"SpaCy model {language}_core_web_sm not found. Downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", f"{language}_core_web_sm"], 
                              check=True)
                self.nlp = spacy.load(f"{language}_core_web_sm")
        
        # Initialize NLTK components if not using spaCy
        if not use_spacy:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
            self.lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Handle None or empty string
        if text is None or text == "":
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Clean text first
        text = self.clean_text(text)
        
        if not text:
            return []
        
        if self.use_spacy:
            # Use spaCy for tokenization, stopword removal, and lemmatization
            doc = self.nlp(text)
            
            if self.remove_stopwords:
                if self.lemmatize:
                    tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.strip()]
                else:
                    tokens = [token.text for token in doc if not token.is_stop and token.text.strip()]
            else:
                if self.lemmatize:
                    tokens = [token.lemma_ for token in doc if token.text.strip()]
                else:
                    tokens = [token.text for token in doc if token.text.strip()]
        else:
            # Use NLTK for tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords if needed
            if self.remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Lemmatize if needed
            if self.lemmatize and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of named entities with type
        """
        if not self.use_spacy:
            raise ValueError("Named entity extraction requires spaCy")
        
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "type": ent.label_
            }
            for ent in doc.ents
        ]
        
        return entities
    
    def extract_features(
        self,
        texts: Union[str, List[str]],
        method: str = "tfidf",
        max_features: int = 1000
    ) -> np.ndarray:
        """
        Extract numerical features from text.
        
        Args:
            texts: Input text or list of texts
            method: Feature extraction method (tfidf, count, or hash)
            max_features: Maximum number of features
            
        Returns:
            Array of numerical features
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean and tokenize texts
        tokenized_texts = [self.tokenize(text) for text in texts]
        
        # Join tokens back to strings for sklearn vectorizers
        processed_texts = [" ".join(tokens) for tokens in tokenized_texts]
        
        if method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=max_features)
            features = vectorizer.fit_transform(processed_texts).toarray()
        elif method == "count":
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=max_features)
            features = vectorizer.fit_transform(processed_texts).toarray()
        elif method == "hash":
            from sklearn.feature_extraction.text import HashingVectorizer
            vectorizer = HashingVectorizer(n_features=max_features)
            features = vectorizer.fit_transform(processed_texts).toarray()
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")
        
        return features