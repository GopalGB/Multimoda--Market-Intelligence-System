# data/preprocessing/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
import datetime
import re
from .text_preprocessor import TextPreprocessor
from .image_preprocessor import ImagePreprocessor


class FeatureEngineer:
    """
    Feature engineering for audience intelligence data.
    
    This class provides methods to:
    - Create combined features from multiple data sources
    - Extract temporal features from timestamps
    - Generate interaction features
    - Normalize and scale features
    - Select important features
    - Reduce dimensionality
    
    It works with multiple data types including structured data, text,
    and visual content to create features optimized for engagement prediction.
    """
    def __init__(
        self,
        text_preprocessor: Optional[TextPreprocessor] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        scaler: str = "standard",  # "standard", "minmax", "robust", or None
        feature_selection: Optional[str] = None,  # "variance", "kbest", or None
        selection_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the feature engineer.
        
        Args:
            text_preprocessor: Text preprocessor for text features
            image_preprocessor: Image preprocessor for image features
            scaler: Type of scaling to apply
            feature_selection: Type of feature selection to apply
            selection_params: Parameters for feature selection
        """
        # Initialize preprocessors
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        
        # Initialize scalers
        self.scaler_type = scaler
        self.scaler = None
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            
        # Initialize feature selection
        self.feature_selection_type = feature_selection
        self.feature_selector = None
        self.selection_params = selection_params or {}
        
        if feature_selection == "variance":
            threshold = self.selection_params.get("threshold", 0.01)
            self.feature_selector = VarianceThreshold(threshold=threshold)
        elif feature_selection == "kbest":
            k = self.selection_params.get("k", 10)
            score_func = self.selection_params.get("score_func", f_regression)
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            
        # Feature tracking
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.engineered_features = []
        self.text_features = []
        self.visual_features = []
        
        # Dimensionality reduction
        self.dim_reducer = None
        
    def fit(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline to data.
        
        Args:
            data: DataFrame with raw data
            target_col: Name of target column (if any)
            
        Returns:
            Self for method chaining
        """
        # Track all feature names
        self.feature_names = list(data.columns)
        
        # Split features by type
        self.categorical_features = []
        self.numerical_features = []
        
        for col in data.columns:
            if col == target_col:
                continue
                
            if data[col].dtype in ['object', 'category', 'bool']:
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)
        
        # Fit scaler on numerical features if needed
        if self.scaler and self.numerical_features:
            numeric_data = data[self.numerical_features].select_dtypes(include=['number'])
            # Replace infinities and handle missing values for scaling
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.scaler.fit(numeric_data)
            
        # Fit feature selector if needed
        if self.feature_selector and target_col is not None:
            if isinstance(self.feature_selector, SelectKBest):
                # For supervised selection, we need the target
                X = self._prepare_data_for_selection(data.drop(columns=[target_col]))
                y = data[target_col]
                self.feature_selector.fit(X, y)
            else:
                # For unsupervised selection like variance threshold
                X = self._prepare_data_for_selection(data.drop(columns=[target_col] if target_col else []))
                self.feature_selector.fit(X)
                
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        add_temporal_features: bool = False,
        add_interaction_features: bool = False
    ) -> pd.DataFrame:
        """
        Transform data using the fitted feature engineering pipeline.
        
        Args:
            data: DataFrame with raw data
            add_temporal_features: Whether to add temporal features
            add_interaction_features: Whether to add interaction features
            
        Returns:
            DataFrame with engineered features
        """
        result = data.copy()
        
        # Add temporal features if requested
        if add_temporal_features:
            result = self.add_temporal_features(result)
            
        # Add interaction features if requested
        if add_interaction_features:
            result = self.add_interaction_features(result)
            
        # Scale numerical features if needed
        if self.scaler and self.numerical_features:
            numeric_cols = [col for col in self.numerical_features if col in result.columns]
            if numeric_cols:
                numeric_data = result[numeric_cols].select_dtypes(include=['number'])
                # Replace infinities and handle missing values for scaling
                numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Apply scaling
                scaled_data = self.scaler.transform(numeric_data)
                result[numeric_cols] = scaled_data
        
        # Apply feature selection if needed
        if self.feature_selector:
            X = self._prepare_data_for_selection(result)
            selected_data = self.feature_selector.transform(X)
            
            # Get selected feature names for column names
            if hasattr(self.feature_selector, 'get_support'):
                # For SelectKBest and VarianceThreshold
                feature_mask = self.feature_selector.get_support()
                selected_features = X.columns[feature_mask].tolist()
                
                # Create new DataFrame with selected features
                result = pd.DataFrame(selected_data, index=result.index, columns=selected_features)
            else:
                # For other feature selectors without get_support
                result = pd.DataFrame(selected_data, index=result.index)
        
        return result
    
    def _prepare_data_for_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for feature selection by handling categorical features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Prepared DataFrame
        """
        # Make a copy to avoid modifying original
        result = data.copy()
        
        # Handle categorical features if any
        cat_cols = [col for col in self.categorical_features if col in result.columns]
        if cat_cols:
            # One-hot encode categorical features
            result = pd.get_dummies(result, columns=cat_cols, drop_first=True)
        
        return result
    
    def add_temporal_features(
        self,
        data: pd.DataFrame,
        date_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add temporal features from date columns.
        
        Args:
            data: Input DataFrame
            date_columns: List of date columns to process (None for auto-detection)
            
        Returns:
            DataFrame with added temporal features
        """
        result = data.copy()
        
        # Auto-detect date columns if not specified
        if date_columns is None:
            date_columns = []
            for col in result.columns:
                # Check if column has datetime type
                if pd.api.types.is_datetime64_any_dtype(result[col]):
                    date_columns.append(col)
                # Try to infer datetime from string columns
                elif result[col].dtype == 'object':
                    try:
                        pd.to_datetime(result[col], errors='raise')
                        date_columns.append(col)
                    except:
                        pass
        
        # Process each date column
        for col in date_columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                try:
                    result[col] = pd.to_datetime(result[col])
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {str(e)}")
                    continue
            
            # Extract date components
            col_prefix = f"{col}_"
            result[col_prefix + 'year'] = result[col].dt.year
            result[col_prefix + 'month'] = result[col].dt.month
            result[col_prefix + 'day'] = result[col].dt.day
            result[col_prefix + 'dayofweek'] = result[col].dt.dayofweek
            result[col_prefix + 'quarter'] = result[col].dt.quarter
            
            # Add cyclic features for time components
            if result[col].dt.time.nunique() > 1:  # Only if there's time variation
                result[col_prefix + 'hour'] = result[col].dt.hour
                
                # Add cyclic encoding for hour (values between -1 and 1)
                hours_in_day = 24
                result[col_prefix + 'hour_sin'] = np.sin(2 * np.pi * result[col].dt.hour / hours_in_day)
                result[col_prefix + 'hour_cos'] = np.cos(2 * np.pi * result[col].dt.hour / hours_in_day)
            
            # Add cyclic encoding for day of week (values between -1 and 1)
            days_in_week = 7
            result[col_prefix + 'day_of_week_sin'] = np.sin(2 * np.pi * result[col].dt.dayofweek / days_in_week)
            result[col_prefix + 'day_of_week_cos'] = np.cos(2 * np.pi * result[col].dt.dayofweek / days_in_week)
            
            # Add cyclic encoding for month (values between -1 and 1)
            months_in_year = 12
            result[col_prefix + 'month_sin'] = np.sin(2 * np.pi * result[col].dt.month / months_in_year)
            result[col_prefix + 'month_cos'] = np.cos(2 * np.pi * result[col].dt.month / months_in_year)
            
            # Add is_weekend feature
            result[col_prefix + 'is_weekend'] = result[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Track engineered features
            self.engineered_features.extend([
                col_prefix + suffix for suffix in [
                    'year', 'month', 'day', 'dayofweek', 'quarter',
                    'hour', 'hour_sin', 'hour_cos',
                    'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos', 'is_weekend'
                ]
            ])
        
        return result
    
    def add_interaction_features(
        self,
        data: pd.DataFrame,
        interaction_cols: Optional[List[Tuple[str, str]]] = None,
        methods: List[str] = ['multiply', 'divide', 'add', 'subtract'],
        max_interactions: int = 10
    ) -> pd.DataFrame:
        """
        Add interaction features between numerical columns.
        
        Args:
            data: Input DataFrame
            interaction_cols: List of column pairs to interact (None for auto detection)
            methods: List of interaction methods to apply
            max_interactions: Maximum number of interactions to create
            
        Returns:
            DataFrame with added interaction features
        """
        result = data.copy()
        numerical_cols = result.select_dtypes(include=['number']).columns.tolist()
        
        # Generate column pairs if not specified
        if interaction_cols is None:
            import itertools
            # Use top correlated features if we have enough numerical columns
            if len(numerical_cols) >= 2:
                # Calculate correlations
                corr_matrix = result[numerical_cols].corr().abs()
                
                # Get top correlated pairs
                pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        correlation = corr_matrix.iloc[i, j]
                        pairs.append((col_i, col_j, correlation))
                
                # Sort by correlation and take top pairs
                pairs.sort(key=lambda x: x[2], reverse=True)
                interaction_cols = [(pair[0], pair[1]) for pair in pairs[:max_interactions]]
            else:
                # Fall back to all combinations if we don't have enough columns
                interaction_cols = list(itertools.combinations(numerical_cols, 2))
                # Limit to max_interactions
                interaction_cols = interaction_cols[:max_interactions]
        
        # Create interaction features
        new_cols = []
        for col1, col2 in interaction_cols:
            if col1 in result.columns and col2 in result.columns:
                # Check if columns are numeric
                if (result[col1].dtype.kind in 'fc') and (result[col2].dtype.kind in 'fc'):
                    # Apply selected methods
                    if 'multiply' in methods:
                        col_name = f"{col1}_{col2}_product"
                        result[col_name] = result[col1] * result[col2]
                        new_cols.append(col_name)
                    
                    if 'divide' in methods:
                        col_name = f"{col1}_div_{col2}"
                        # Avoid division by zero
                        result[col_name] = result[col1] / (result[col2] + 1e-10)
                        new_cols.append(col_name)
                    
                    if 'add' in methods:
                        col_name = f"{col1}_{col2}_sum"
                        result[col_name] = result[col1] + result[col2]
                        new_cols.append(col_name)
                    
                    if 'subtract' in methods:
                        col_name = f"{col1}_minus_{col2}"
                        result[col_name] = result[col1] - result[col2]
                        new_cols.append(col_name)
        
        # Track engineered features
        self.engineered_features.extend(new_cols)
        
        return result
    
    def extract_text_features(
        self,
        data: pd.DataFrame,
        text_column: str,
        methods: List[str] = ['embeddings', 'linguistic'],
        embedding_col_prefix: str = 'text_emb_',
        max_dims: int = 10
    ) -> pd.DataFrame:
        """
        Extract features from text data.
        
        Args:
            data: Input DataFrame
            text_column: Column containing text data
            methods: List of feature extraction methods
            embedding_col_prefix: Prefix for embedding columns
            max_dims: Maximum dimensions for embeddings
            
        Returns:
            DataFrame with added text features
        """
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        result = data.copy()
        new_cols = []
        
        # Process text
        texts = result[text_column].fillna("").tolist()
        
        # Extract embeddings
        if 'embeddings' in methods:
            # Get text embeddings
            embeddings = self.text_preprocessor.encode_text(texts).cpu().numpy()
            
            # Limit dimensions
            dims_to_use = min(embeddings.shape[1], max_dims)
            
            # Add embedding features
            for i in range(dims_to_use):
                col_name = f"{embedding_col_prefix}{i}"
                result[col_name] = embeddings[:, i]
                new_cols.append(col_name)
                
            # Track text features
            self.text_features.extend(new_cols)
        
        # Extract linguistic features
        if 'linguistic' in methods:
            # For each text, extract linguistic features
            linguistic_features = []
            
            for text in texts:
                features = {}
                
                # Get basic linguistic features
                if hasattr(self.text_preprocessor, 'extract_linguistic_features'):
                    ling_feats = self.text_preprocessor.extract_linguistic_features(text)
                    
                    # Flatten the features if needed
                    if isinstance(ling_feats, dict):
                        for feat_type, feat_values in ling_feats.items():
                            if isinstance(feat_values, np.ndarray):
                                # First feature from each type
                                features[f"ling_{feat_type}"] = feat_values[0][0]
                            else:
                                features[f"ling_{feat_type}"] = feat_values
                    elif isinstance(ling_feats, np.ndarray):
                        # Use first few features
                        for i in range(min(ling_feats.shape[1], 5)):
                            features[f"ling_feat_{i}"] = ling_feats[0, i]
                
                # Add engagement features
                if hasattr(self.text_preprocessor, 'extract_engagement_features'):
                    eng_feats = self.text_preprocessor.extract_engagement_features(text)
                    features.update({
                        f"text_engagement_{k}": v for k, v in eng_feats.items()
                    })
                
                linguistic_features.append(features)
            
            # Convert to DataFrame and join with result
            if linguistic_features:
                ling_df = pd.DataFrame(linguistic_features, index=data.index)
                
                # Add to result
                for col in ling_df.columns:
                    result[col] = ling_df[col]
                    new_cols.append(col)
                
                # Track text features
                self.text_features.extend(ling_df.columns.tolist())
        
        return result
    
    def reduce_dimensionality(
        self,
        data: pd.DataFrame,
        n_components: int = 10,
        method: str = 'pca',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Reduce dimensionality of numeric features.
        
        Args:
            data: Input DataFrame
            n_components: Number of components to keep
            method: Dimensionality reduction method ('pca', 'kernel_pca', 'tsne')
            columns: Columns to use (None for all numeric)
            
        Returns:
            DataFrame with reduced features
        """
        # Select columns to use for dimensionality reduction
        if columns is None:
            # Use all numeric columns
            columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # Create dimensionality reducer if needed
        if method == 'pca':
            from sklearn.decomposition import PCA
            self.dim_reducer = PCA(n_components=n_components)
        elif method == 'kernel_pca':
            from sklearn.decomposition import KernelPCA
            self.dim_reducer = KernelPCA(n_components=n_components, kernel='rbf')
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            self.dim_reducer = TSNE(n_components=n_components, perplexity=30)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        # Apply dimensionality reduction
        numeric_data = data[columns].select_dtypes(include=['number'])
        reduced_data = self.dim_reducer.fit_transform(numeric_data)
        
        # Create result DataFrame
        result = data.copy()
        
        # Add reduced features
        for i in range(reduced_data.shape[1]):
            col_name = f"{method}_{i}"
            result[col_name] = reduced_data[:, i]
        
        return result
    
    def extract_engagement_features(
        self,
        data: pd.DataFrame,
        content_columns: Optional[Dict[str, str]] = None,
        temporal_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract features specifically optimized for audience engagement prediction.
        
        Args:
            data: Input DataFrame
            content_columns: Dict mapping column types to column names
            temporal_col: Column with temporal data
            
        Returns:
            DataFrame with engagement features
        """
        result = data.copy()
        
        # Default columns if not specified
        if content_columns is None:
            content_columns = {}
        
        # Add temporal features if temporal column provided
        if temporal_col:
            # Add basic temporal features
            result = self.add_temporal_features(result, [temporal_col])
            
            # Add specialized "prime time" feature if hour is available
            hour_col = f"{temporal_col}_hour"
            if hour_col in result.columns:
                # Define prime time as 7-10 PM (19-22)
                result['is_prime_time'] = ((result[hour_col] >= 19) & 
                                          (result[hour_col] <= 22)).astype(int)
                self.engineered_features.append('is_prime_time')
        
        # Process text content
        if 'text' in content_columns:
            text_col = content_columns['text']
            if text_col in result.columns:
                result = self.extract_text_features(
                    result, text_col, 
                    methods=['embeddings', 'linguistic']
                )
        
        # Add cross-feature for day of week and content type if available
        if 'content_type' in result.columns and temporal_col:
            day_col = f"{temporal_col}_dayofweek"
            if day_col in result.columns:
                # Create cross feature for content type and day of week
                result['content_day'] = result['content_type'] + '_' + result[day_col].astype(str)
                self.engineered_features.append('content_day')
        
        # Add historical engagement features if available
        if 'user_id' in result.columns and 'engagement' in result.columns:
            # Calculate historical engagement rate per user
            user_stats = result.groupby('user_id')['engagement'].agg(['mean', 'count']).reset_index()
            user_stats.columns = ['user_id', 'user_avg_engagement', 'user_content_count']
            
            # Merge back to result
            result = result.merge(user_stats, on='user_id', how='left')
            
            # Fill NaN values for new users
            result['user_avg_engagement'] = result['user_avg_engagement'].fillna(result['engagement'].mean())
            result['user_content_count'] = result['user_content_count'].fillna(0)
            
            # Track engineered features
            self.engineered_features.extend(['user_avg_engagement', 'user_content_count'])
        
        return result
    
    def generate_batch_features(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate features for a batch of data based on configuration.
        
        Args:
            data: Input DataFrame
            config: Configuration dictionary
            
        Returns:
            DataFrame with generated features
        """
        result = data.copy()
        
        # Apply feature generation based on config
        if config.get('add_temporal', False):
            date_cols = config.get('date_columns')
            result = self.add_temporal_features(result, date_cols)
        
        if config.get('add_interactions', False):
            interaction_cols = config.get('interaction_columns')
            methods = config.get('interaction_methods', ['multiply', 'divide', 'add', 'subtract'])
            result = self.add_interaction_features(result, interaction_cols, methods)
        
        if config.get('text_features', False):
            text_col = config.get('text_column')
            if text_col and text_col in result.columns:
                methods = config.get('text_methods', ['embeddings'])
                result = self.extract_text_features(result, text_col, methods)
        
        if config.get('reduce_dims', False):
            n_components = config.get('n_components', 10)
            method = config.get('dim_method', 'pca')
            columns = config.get('dim_columns')
            result = self.reduce_dimensionality(result, n_components, method, columns)
        
        if config.get('engagement_features', False):
            content_columns = config.get('content_columns')
            temporal_col = config.get('temporal_column')
            result = self.extract_engagement_features(result, content_columns, temporal_col)
        
        # Apply scaling if configured
        if config.get('apply_scaling', False) and self.scaler:
            numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
            result[numeric_cols] = self.scaler.transform(result[numeric_cols])
        
        return result