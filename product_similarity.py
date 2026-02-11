
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class ProductSimilarity:
    """Calculate product similarity using collaborative and content-based methods"""
    
    def __init__(self, df_reviews, df_products):
        """
        Initialize similarity calculator
        
        Args:
            df_reviews: DataFrame of product reviews
            df_products: DataFrame of product metadata
        """
        self.df_reviews = df_reviews
        self.df_products = df_products
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Cache for performance
        self._collaborative_cache = {}
        self._content_cache = {}
    
    def calculate_collaborative_similarity(self, target_asin, top_n=50):
        """
        Calculate collaborative filtering similarity based on co-review patterns
        "Users who reviewed X also reviewed Y"
        
        Args:
            target_asin: Product ASIN to find similar products for
            top_n: Number of similar products to return
            
        Returns:
            List of (asin, similarity_score) tuples, sorted by similarity
        """
        # Check cache
        if target_asin in self._collaborative_cache:
            return self._collaborative_cache[target_asin][:top_n]
        
        # Get users who reviewed the target product
        target_reviews = self.df_reviews[self.df_reviews['asin'] == target_asin]
        if len(target_reviews) == 0:
            return []
        
        target_users = set(target_reviews['user_id'].unique())
        
        # Find other products reviewed by these users
        user_reviews = self.df_reviews[self.df_reviews['user_id'].isin(target_users)]
        
        # Count co-reviews and calculate similarity
        product_scores = defaultdict(lambda: {'count': 0, 'ratings': []})
        
        for _, row in user_reviews.iterrows():
            asin = row['asin']
            if asin != target_asin:
                product_scores[asin]['count'] += 1
                product_scores[asin]['ratings'].append(row['rating'])
        
        # Calculate similarity score
        similarities = []
        target_avg_rating = target_reviews['rating'].mean()
        
        for asin, data in product_scores.items():
            if data['count'] < 1:  # Require at least 1 co-review (lowered for sparse data)
                continue
            
            # Frequency score (normalized by target product's review count)
            frequency_score = min(data['count'] / len(target_users), 1.0)
            
            # Rating correlation score
            avg_rating = np.mean(data['ratings'])
            rating_similarity = 1 - abs(target_avg_rating - avg_rating) / 5.0
            
            # Combined score
            similarity = 0.7 * frequency_score + 0.3 * rating_similarity
            similarities.append((asin, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Cache result
        self._collaborative_cache[target_asin] = similarities
        
        return similarities[:top_n]
    
    def calculate_content_similarity(self, target_asin, top_n=50):
        """
        Calculate content-based similarity using product metadata
        
        Args:
            target_asin: Product ASIN to find similar products for
            top_n: Number of similar products to return
            
        Returns:
            List of (asin, similarity_score) tuples, sorted by similarity
        """
        # Check cache
        if target_asin in self._content_cache:
            return self._content_cache[target_asin][:top_n]
        
        # Get target product
        target_product = self.df_products[self.df_products['asin'] == target_asin]
        if target_product.empty:
            return []
        
        target_product = target_product.iloc[0]
        
        # Prepare features for all products
        similarities = []
        
        for _, product in self.df_products.iterrows():
            if product['asin'] == target_asin:
                continue
            
            similarity_scores = []
            
            # 1. Category similarity (40%)
            category_sim = self._calculate_category_similarity(target_product, product)
            similarity_scores.append(('category', category_sim, 0.40))
            
            # 2. Title similarity (30%)
            title_sim = self._calculate_title_similarity(target_product, product)
            similarity_scores.append(('title', title_sim, 0.30))
            
            # 3. Price similarity (20%)
            price_sim = self._calculate_price_similarity(target_product, product)
            similarity_scores.append(('price', price_sim, 0.20))
            
            # 4. Feature similarity (10%)
            feature_sim = self._calculate_feature_similarity(target_product, product)
            similarity_scores.append(('feature', feature_sim, 0.10))
            
            # Calculate weighted similarity
            total_similarity = sum(score * weight for _, score, weight in similarity_scores)
            
            similarities.append((product['asin'], total_similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Cache result
        self._content_cache[target_asin] = similarities
        
        return similarities[:top_n]
    
    def _calculate_category_similarity(self, product1, product2):
        """Calculate similarity based on category"""
        try:
            cat1 = product1.get('main_category', '')
            cat2 = product2.get('main_category', '')
            
            if pd.isna(cat1) or pd.isna(cat2):
                return 0.5
            
            # Exact match
            if cat1 == cat2:
                return 1.0
            
            # Partial match (check if one is substring of other)
            cat1_lower = str(cat1).lower()
            cat2_lower = str(cat2).lower()
            if cat1_lower in cat2_lower or cat2_lower in cat1_lower:
                return 0.7
            
            return 0.0
        except Exception:
            return 0.5
    
    def _calculate_title_similarity(self, product1, product2):
        """Calculate similarity based on product title using TF-IDF"""
        try:
            title1 = str(product1.get('title', ''))
            title2 = str(product2.get('title', ''))
            
            if not title1 or not title2:
                return 0.0
            
            # Use TF-IDF
            tfidf_matrix = self.tfidf.fit_transform([title1, title2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception:
            return 0.0
    
    def _calculate_price_similarity(self, product1, product2):
        """Calculate similarity based on price range"""
        try:
            price1 = pd.to_numeric(product1.get('price', 0), errors='coerce')
            price2 = pd.to_numeric(product2.get('price', 0), errors='coerce')
            
            if pd.isna(price1) or pd.isna(price2) or price1 == 0 or price2 == 0:
                return 0.5
            
            # Calculate relative price difference
            price_diff = abs(price1 - price2) / max(price1, price2)
            
            # Convert to similarity (closer prices = higher similarity)
            similarity = 1.0 - min(price_diff, 1.0)
            
            return similarity
        except Exception:
            return 0.5
    
    def _calculate_feature_similarity(self, product1, product2):
        """Calculate similarity based on product features"""
        try:
            features1 = product1.get('features', [])
            features2 = product2.get('features', [])
            
            if not features1 or not features2:
                return 0.5
            
            # Convert to sets for Jaccard similarity
            set1 = set(str(f).lower() for f in features1 if pd.notna(f))
            set2 = set(str(f).lower() for f in features2 if pd.notna(f))
            
            if not set1 or not set2:
                return 0.5
            
            # Jaccard similarity
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.5
    
    def calculate_hybrid_similarity(self, target_asin, top_n=50, collaborative_weight=0.6, content_weight=0.4):
        """
        Calculate hybrid similarity combining collaborative and content-based methods
        
        Args:
            target_asin: Product ASIN to find similar products for
            top_n: Number of similar products to return
            collaborative_weight: Weight for collaborative filtering (default 0.6)
            content_weight: Weight for content-based filtering (default 0.4)
            
        Returns:
            List of (asin, similarity_score, method_breakdown) tuples
        """
        # Get collaborative similarities
        collab_similarities = dict(self.calculate_collaborative_similarity(target_asin, top_n=100))
        
        # Get content-based similarities
        content_similarities = dict(self.calculate_content_similarity(target_asin, top_n=100))
        
        # Combine scores
        all_asins = set(collab_similarities.keys()) | set(content_similarities.keys())
        
        hybrid_scores = []
        for asin in all_asins:
            collab_score = collab_similarities.get(asin, 0.0)
            content_score = content_similarities.get(asin, 0.0)
            
            # Hybrid score
            hybrid_score = collaborative_weight * collab_score + content_weight * content_score
            
            hybrid_scores.append((
                asin,
                hybrid_score,
                {
                    'collaborative': collab_score,
                    'content': content_score
                }
            ))
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_scores[:top_n]
