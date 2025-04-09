# models/text/text_features.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, RobertaTokenizer, RobertaModel
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
import string
from enum import Enum

class TextFeatureTypes(Enum):
    """Enumeration of available text feature types."""
    EMBEDDINGS = "embeddings"
    LINGUISTIC = "linguistic"
    SENTIMENT = "sentiment"
    KEYWORDS = "keywords"
    ENTITIES = "entities"
    ENGAGEMENT = "engagement"
    READABILITY = "readability"
    ALL = "all"

class TextFeatureExtractor:
    """
    Extract and process textual features from content for audience analysis.
    
    This class provides methods to:
    - Generate text embeddings for semantic understanding
    - Extract keywords and topics
    - Analyze sentiment and emotion
    - Extract named entities and content attributes
    - Generate linguistic features related to audience engagement
    """
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: Optional[str] = None,
        pooling_strategy: str = "cls",  # cls, mean, max
        use_spacy: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text feature extractor.
        
        Args:
            model_name: Transformer model name or path
            device: Compute device ('cpu', 'cuda', or None for auto)
            pooling_strategy: Strategy for sentence embedding pooling
            use_spacy: Whether to use spaCy for additional NLP features
            cache_dir: Directory to cache models
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load tokenizer and model
        self.model_name = model_name
        
        if "roberta" in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set to evaluation mode for inference
        self.model.eval()
        
        # Set pooling strategy
        self.pooling_strategy = pooling_strategy
        
        # Load spaCy model if enabled
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Download if not available
                import subprocess
                print("Downloading spaCy model...")
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                              check=True)
                self.nlp = spacy.load("en_core_web_sm")
        
        # Sentiment words for lexicon-based sentiment analysis
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        
        # Linguistic feature extractors
        self.tfidf_vectorizer = None  # Lazy initialization
        
        # Track model dimension
        self.embedding_dim = self.model.config.hidden_size
    
    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """
        Load basic sentiment lexicon for supplementary sentiment analysis.
        
        Returns:
            Dictionary mapping words to sentiment scores
        """
        # Basic lexicon for demonstration, would be replaced with a more comprehensive one
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "outstanding", "brilliant", "terrific", "awesome", "superb", "delightful",
            "enjoyable", "engaging", "entertaining", "compelling", "impressive", "exciting",
            "innovative", "creative", "insightful"
        ]
        
        negative_words = [
            "bad", "poor", "terrible", "awful", "horrible", "dreadful",
            "disappointing", "mediocre", "boring", "dull", "uninteresting", "tedious",
            "confusing", "frustrating", "annoying", "irritating", "inadequate", "flawed",
            "repetitive", "derivative", "shallow"
        ]
        
        # Create lexicon with positive words having score 1.0 and negative -1.0
        lexicon = {word: 1.0 for word in positive_words}
        lexicon.update({word: -1.0 for word in negative_words})
        
        return lexicon
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before feature extraction.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Handle special characters
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\"\'&]', '', text)
        
        return text
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Encode text into embeddings.
        
        Args:
            texts: Input text or list of texts
            normalize: Whether to L2 normalize embeddings
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Text embeddings tensor and optionally hidden states
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts = [self._preprocess_text(text) for text in texts]
        
        # Process in batches
        all_embeddings = []
        all_hidden_states = [] if return_hidden_states else None
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=return_hidden_states,
                    return_dict=True
                )
            
            # Apply pooling strategy
            if self.pooling_strategy == "cls":
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooling_strategy == "mean":
                # Mean pooling over tokens
                attention_mask = inputs["attention_mask"]
                batch_embeddings = self._mean_pooling(
                    outputs.last_hidden_state, attention_mask
                )
            elif self.pooling_strategy == "max":
                # Max pooling over tokens
                batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
            else:
                raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
            
            # Normalize if requested
            if normalize:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # Add to results
            all_embeddings.append(batch_embeddings)
            
            # Add hidden states if requested
            if return_hidden_states:
                all_hidden_states.append(outputs.hidden_states)
        
        # Combine batches
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if return_hidden_states:
            # Handle hidden states format for multiple batches
            combined_hidden_states = []
            for layer_idx in range(len(all_hidden_states[0])):
                layer_states = torch.cat(
                    [batch_hidden[layer_idx] for batch_hidden in all_hidden_states],
                    dim=0
                )
                combined_hidden_states.append(layer_states)
            
            return embeddings, combined_hidden_states
        
        return embeddings
    
    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings.
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Mean-pooled embeddings [batch_size, hidden_size]
        """
        # Expand mask to same dimension as embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings weighted by mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum mask to get actual token count
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Average embeddings over tokens
        pooled = sum_embeddings / sum_mask
        
        return pooled
    
    def extract_linguistic_features(
        self,
        texts: Union[str, List[str]],
        feature_types: List[str] = ["lexical", "syntactic", "readability"]
    ) -> Dict[str, np.ndarray]:
        """
        Extract linguistic features relevant to audience engagement.
        
        Args:
            texts: Input text or list of texts
            feature_types: Types of features to extract
            
        Returns:
            Dictionary mapping feature types to feature arrays
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts = [self._preprocess_text(text) for text in texts]
        
        # Initialize results
        results = {}
        
        # Extract lexical features
        if "lexical" in feature_types:
            lexical_features = self._extract_lexical_features(texts)
            results["lexical"] = lexical_features
        
        # Extract syntactic features
        if "syntactic" in feature_types and self.nlp is not None:
            syntactic_features = self._extract_syntactic_features(texts)
            results["syntactic"] = syntactic_features
        
        # Extract readability features
        if "readability" in feature_types:
            readability_features = self._extract_readability_features(texts)
            results["readability"] = readability_features
        
        return results
    
    def _extract_lexical_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract lexical features related to vocabulary usage.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Array of lexical features [num_texts, num_features]
        """
        features = []
        
        for text in texts:
            # Tokenize into words
            if self.nlp:
                doc = self.nlp(text)
                words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
            else:
                # Simple word tokenization without spaCy
                words = [word.lower() for word in re.findall(r'\b\w+\b', text)]
            
            # Calculate lexical features
            num_words = len(words)
            num_unique_words = len(set(words))
            lexical_diversity = num_unique_words / max(num_words, 1)  # Type-token ratio
            
            # Word length statistics
            word_lengths = [len(word) for word in words]
            avg_word_length = np.mean(word_lengths) if word_lengths else 0
            max_word_length = max(word_lengths) if word_lengths else 0
            
            # N-grams
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
            num_bigrams = len(bigrams)
            num_unique_bigrams = len(set(bigrams))
            bigram_diversity = num_unique_bigrams / max(num_bigrams, 1)
            
            # Content word ratio (nouns, verbs, adjectives, adverbs)
            if self.nlp:
                content_words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
                content_word_ratio = len(content_words) / max(num_words, 1)
            else:
                content_word_ratio = 0.5  # Default if spaCy not available
            
            # Assemble feature vector
            text_features = [
                num_words, 
                num_unique_words,
                lexical_diversity,
                avg_word_length,
                max_word_length,
                bigram_diversity,
                content_word_ratio
            ]
            
            features.append(text_features)
        
        return np.array(features)
    
    def _extract_syntactic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract syntactic features related to sentence structure.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Array of syntactic features [num_texts, num_features]
        """
        if self.nlp is None:
            raise ValueError("spaCy model required for syntactic feature extraction")
        
        features = []
        
        for text in texts:
            doc = self.nlp(text)
            
            # Sentence information
            sentences = list(doc.sents)
            num_sentences = len(sentences)
            
            # Calculate average sentence length
            sentence_lengths = [len(sent) for sent in sentences]
            avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
            
            # Calculate syntactic complexity features
            num_nouns = len([token for token in doc if token.pos_ == "NOUN"])
            num_verbs = len([token for token in doc if token.pos_ == "VERB"])
            num_adjectives = len([token for token in doc if token.pos_ == "ADJ"])
            num_adverbs = len([token for token in doc if token.pos_ == "ADV"])
            
            # Noun-verb ratio (higher values indicate more descriptive text)
            noun_verb_ratio = num_nouns / max(num_verbs, 1)
            
            # Modifier ratio (higher values indicate more descriptive text)
            modifier_ratio = (num_adjectives + num_adverbs) / max(num_nouns + num_verbs, 1)
            
            # Parse tree depth (approximation of syntactic complexity)
            max_depth = max([token._.get_depth() if hasattr(token._, 'get_depth') else 0 
                            for token in doc] or [0])
            
            # Named entities
            num_entities = len(doc.ents)
            
            # Assemble feature vector
            text_features = [
                num_sentences,
                avg_sentence_length,
                noun_verb_ratio,
                modifier_ratio,
                max_depth,
                num_entities
            ]
            
            features.append(text_features)
        
        return np.array(features)
    
    def _extract_readability_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract readability features related to text complexity.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Array of readability features [num_texts, num_features]
        """
        features = []
        
        for text in texts:
            # Basic tokenization
            if self.nlp:
                doc = self.nlp(text)
                sentences = list(doc.sents)
                words = [token.text for token in doc if not token.is_punct and not token.is_space]
            else:
                # Simple sentence splitting without spaCy
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                words = [word for word in re.findall(r'\b\w+\b', text)]
            
            # Calculate basic statistics
            num_chars = len(text)
            num_words = len(words)
            num_sentences = len(sentences)
            
            # Calculate average word length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Calculate average sentence length
            avg_sentence_length = num_words / max(num_sentences, 1)
            
            # Approximate syllable count (heuristic)
            syllable_count = 0
            multi_syllable_words = 0
            
            for word in words:
                word = word.lower()
                if word.endswith(('es', 'ed')):
                    word = word[:-2]
                elif word.endswith('e'):
                    word = word[:-1]
                
                vowels = 'aeiouy'
                vowel_groups = re.findall(f'[{vowels}]+', word)
                word_syllables = len(vowel_groups)
                
                if word_syllables > 2:
                    multi_syllable_words += 1
                    
                syllable_count += max(1, word_syllables)
            
            # Flesch-Kincaid Grade Level
            fk_grade = 0.39 * avg_sentence_length + 11.8 * (syllable_count / max(num_words, 1)) - 15.59
            
            # Flesch Reading Ease
            flesch_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * (syllable_count / max(num_words, 1))
            
            # Percentage of complex words
            complex_word_percentage = multi_syllable_words / max(num_words, 1)
            
            # Assemble feature vector
            text_features = [
                avg_sentence_length,
                avg_word_length,
                fk_grade,
                flesch_ease,
                complex_word_percentage
            ]
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_keywords(
        self,
        text: str,
        top_n: int = 10,
        method: str = "tfidf"
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            method: Keyword extraction method ("tfidf", "yake", "spacy")
            
        Returns:
            List of dictionaries with keywords and scores
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        if method == "tfidf":
            # TF-IDF based keyword extraction
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_df=0.95, min_df=1, stop_words="english"
                )
                # Fit on a single document (will be updated as more texts are processed)
                self.tfidf_vectorizer.fit([text])
            
            # Get TF-IDF scores
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Sort by score
            scores = tfidf_matrix.toarray()[0]
            sorted_indices = np.argsort(scores)[::-1]
            
            # Get top keywords
            keywords = []
            for idx in sorted_indices[:top_n]:
                if scores[idx] > 0:
                    keywords.append({
                        "keyword": feature_names[idx],
                        "score": float(scores[idx])
                    })
            
            return keywords
            
        elif method == "spacy":
            # spaCy based keyword extraction
            if self.nlp is None:
                raise ValueError("spaCy model required for spaCy keyword extraction")
            
            doc = self.nlp(text)
            
            # Count noun chunks and named entities
            counts = Counter()
            for chunk in doc.noun_chunks:
                counts[chunk.text.lower()] += 1
            
            for ent in doc.ents:
                counts[ent.text.lower()] += 2  # Weight entities higher
            
            # Filter out stopwords and punctuation
            keywords = [
                {"keyword": keyword, "score": count}
                for keyword, count in counts.most_common(top_n)
                if keyword not in self.nlp.Defaults.stop_words
                and not all(c in string.punctuation for c in keyword)
                and len(keyword.split()) <= 4  # Limit phrase length
            ]
            
            return keywords
            
        else:
            raise ValueError(f"Unsupported keyword extraction method: {method}")
    
    def analyze_sentiment(
        self,
        text: str,
        method: str = "transformer"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment in text.
        
        Args:
            text: Input text
            method: Sentiment analysis method ("transformer" or "lexicon")
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        if method == "transformer":
            # Use transformer model to predict sentiment
            # For demonstration, use model output with simple processing
            embeddings = self.encode_text(text)
            
            # Project embeddings to sentiment score (approximation)
            logits = torch.nn.Linear(self.embedding_dim, 1).to(self.device)(embeddings)
            score = torch.tanh(logits).item()  # Scale to [-1, 1]
            
            # Map to sentiment categories
            if score < -0.5:
                sentiment = "very_negative"
                sentiment_score = -0.8
            elif score < -0.1:
                sentiment = "negative"
                sentiment_score = -0.4
            elif score < 0.1:
                sentiment = "neutral"
                sentiment_score = 0.0
            elif score < 0.5:
                sentiment = "positive"
                sentiment_score = 0.4
            else:
                sentiment = "very_positive"
                sentiment_score = 0.8
            
            return {
                "sentiment": sentiment,
                "score": sentiment_score,
                "confidence": abs(sentiment_score)
            }
            
        elif method == "lexicon":
            # Use lexicon-based sentiment analysis
            if self.nlp:
                # Use spaCy for tokenization
                doc = self.nlp(text)
                tokens = [token.lemma_.lower() for token in doc 
                         if not token.is_punct and not token.is_space]
            else:
                # Simple tokenization
                tokens = [word.lower() for word in re.findall(r'\b\w+\b', text)]
            
            # Calculate sentiment score
            sentiment_values = [self.sentiment_lexicon.get(token, 0) for token in tokens]
            if sentiment_values:
                sentiment_score = sum(sentiment_values) / len(sentiment_values)
            else:
                sentiment_score = 0
            
            # Map to sentiment categories
            if sentiment_score < -0.5:
                sentiment = "very_negative"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            elif sentiment_score < 0.1:
                sentiment = "neutral"
            elif sentiment_score < 0.5:
                sentiment = "positive"
            else:
                sentiment = "very_positive"
            
            return {
                "sentiment": sentiment,
                "score": sentiment_score,
                "confidence": abs(sentiment_score)
            }
            
        else:
            raise ValueError(f"Unsupported sentiment analysis method: {method}")
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping emotions to scores
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Emotion lexicons (would be expanded in production)
        emotion_lexicons = {
            "joy": ["happy", "joy", "delighted", "pleased", "glad", "excited", "thrilled"],
            "sadness": ["sad", "unhappy", "depressed", "gloomy", "miserable", "disappointed"],
            "anger": ["angry", "furious", "irritated", "annoyed", "enraged", "hostile"],
            "fear": ["afraid", "scared", "frightened", "fearful", "terrified", "anxious"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "startled"],
            "disgust": ["disgusted", "revolted", "repulsed", "appalled", "distaste"],
            "trust": ["trust", "believe", "faith", "confident", "assured", "reliable"],
            "anticipation": ["expect", "anticipate", "await", "look forward", "predict"]
        }
        
        # Tokenize text
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_punct and not token.is_space]
        else:
            tokens = [word.lower() for word in re.findall(r'\b\w+\b', text)]
        
        # Count emotion words
        emotion_scores = {emotion: 0.0 for emotion in emotion_lexicons}
        
        for token in tokens:
            for emotion, words in emotion_lexicons.items():
                if token in words:
                    emotion_scores[emotion] += 1.0
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {emotion: score / total_score 
                             for emotion, score in emotion_scores.items()}
        
        return emotion_scores
    
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of named entity dictionaries
        """
        if self.nlp is None:
            raise ValueError("spaCy model required for named entity extraction")
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def extract_engagement_features(self, text: str) -> Dict[str, float]:
        """
        Extract features specifically related to audience engagement.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of engagement-related features
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Analyze linguistic features
        linguistic_features = self.extract_linguistic_features(
            text, feature_types=["lexical", "readability"]
        )
        
        # Calculate lexical diversity (indicates richness of vocabulary)
        lexical_diversity = linguistic_features["lexical"][0][2]
        
        # Calculate readability (higher scores indicate easier to read content)
        readability = min(100, max(0, linguistic_features["readability"][0][3]))
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Calculate sentiment intensity (regardless of direction)
        sentiment_intensity = abs(sentiment["score"])
        
        # Extract keywords
        keywords = self.extract_keywords(text, method="tfidf" if self.tfidf_vectorizer else "spacy")
        
        # Calculate keyword strength (higher scores indicate more distinctive keywords)
        keyword_strength = sum(kw["score"] for kw in keywords[:5]) if keywords else 0
        
        # Engagement metrics based on text characteristics
        engagement_features = {
            "vocabulary_richness": float(lexical_diversity),
            "readability": float(readability / 100),  # Normalize to [0, 1]
            "sentiment_intensity": float(sentiment_intensity),
            "keyword_strength": float(min(1.0, keyword_strength)),  # Cap at 1.0
            "estimated_engagement": float(
                0.3 * lexical_diversity + 
                0.3 * (readability / 100) + 
                0.2 * sentiment_intensity +
                0.2 * min(1.0, keyword_strength)
            )
        }
        
        return engagement_features
    
    def compute_text_similarity(
        self,
        text1: str,
        text2: str,
        method: str = "cosine"
    ) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score
        """
        # Encode texts
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        # Compute similarity based on method
        if method == "cosine":
            # Cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1)[0].item()
        elif method == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = torch.norm(embedding1 - embedding2, p=2, dim=1)[0].item()
            similarity = 1.0 / (1.0 + distance)  # Transform to [0, 1]
        elif method == "dot":
            # Dot product
            similarity = torch.sum(embedding1 * embedding2, dim=1)[0].item()
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        return similarity
    
    def classify_content(
        self,
        text: str,
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Classify content into predefined categories.
        
        Args:
            text: Input text
            categories: List of category names
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        # Encode text
        text_embedding = self.encode_text(text)
        
        # Encode categories
        category_embeddings = self.encode_text(categories)
        
        # Compute similarities between text and categories
        similarities = F.cosine_similarity(
            text_embedding.unsqueeze(1),
            category_embeddings.unsqueeze(0),
            dim=2
        )[0]
        
        # Create mapping of categories to scores
        category_scores = {
            category: score.item()
            for category, score in zip(categories, similarities)
        }
        
        # Sort by score
        category_scores = dict(sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return category_scores