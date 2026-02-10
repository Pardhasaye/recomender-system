
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TrustworthyRecommender:
    """Trust scoring system for products, users, and sellers"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.mean_price = None
        self.title_col = None
        self.seller_col = None
        
    def user_trust_score(self, df_reviews, user_id):
        """Calculate user trustworthiness score"""
        # Determine user ID column
        uid_col = 'user_id' if 'user_id' in df_reviews.columns else ('reviewerID' if 'reviewerID' in df_reviews.columns else None)
        if not uid_col: return 0.5
        
        user_reviews = df_reviews[df_reviews[uid_col] == user_id]
        if len(user_reviews) == 0: return 0.5
        
        if 'verified_purchase' in user_reviews.columns:
            v_ratio = user_reviews['verified_purchase'].mean()
        else:
            v_ratio = 0.5 # Default if missing
            
        if 'helpful_vote' in user_reviews.columns:
            h_votes = pd.to_numeric(user_reviews['helpful_vote'], errors='coerce').fillna(0)
        elif 'vote' in user_reviews.columns:
            h_votes = pd.to_numeric(user_reviews['vote'], errors='coerce').fillna(0)
        else:
            h_votes = pd.Series([0] * len(user_reviews))
            
        h_ratio = h_votes.mean() / max((h_votes + 1).mean(), 1)
        
        if 'rating' in user_reviews.columns:
            ratings = pd.to_numeric(user_reviews['rating'], errors='coerce')
        elif 'overall' in user_reviews.columns:
            ratings = pd.to_numeric(user_reviews['overall'], errors='coerce')
        else:
            ratings = pd.Series([5.0] * len(user_reviews))
            
        rating_std = ratings.std()
        c_rating = 1 - (rating_std / 2.0) if pd.notna(rating_std) else 1.0
        
        text_len = user_reviews.get('text', '').fillna('').str.len()
        q_text = np.clip(text_len.mean() / max(text_len.median(), 10), 0, 1)
        
        u_trust = 0.3 * v_ratio + 0.25 * h_ratio + 0.2 * c_rating + 0.25 * q_text
        return np.clip(u_trust, 0.0, 1.0)
    
    def product_trust_score(self, df_reviews, df_products, asin):
        """Calculate product trustworthiness score"""
        product_reviews = df_reviews[df_reviews['asin'] == asin]
        if len(product_reviews) == 0: return 0.0
        
        if 'rating' in product_reviews.columns:
            ratings = pd.to_numeric(product_reviews['rating'], errors='coerce')
        elif 'overall' in product_reviews.columns:
            ratings = pd.to_numeric(product_reviews['overall'], errors='coerce')
        else:
            ratings = pd.Series([5.0] * len(product_reviews))
        avg_rating = ratings.mean()
        rating_num = len(product_reviews)
        if 'verified_purchase' in product_reviews.columns:
            v_share = product_reviews['verified_purchase'].mean()
        else:
            v_share = 0.0
        rn_conf = 1 - np.exp(-rating_num / 1000)
        
        # Price abnormality factor
        price_factor = 1.0
        try:
            if self.mean_price and len(df_products) > 0:
                prod_row = df_products[df_products['asin'] == asin]
                if not prod_row.empty:
                    price = float(pd.to_numeric(prod_row.get('price').iloc[0]))
                    if not np.isnan(self.mean_price) and self.mean_price > 0:
                        price_diff = abs(price - self.mean_price) / float(self.mean_price)
                        price_factor = 1.0 - min(price_diff, 1.0)
        except Exception:
            price_factor = 1.0

        # Title-review similarity
        title_sim = 0.0
        try:
            if self.title_col and self.title_col in df_products.columns:
                prod_row = df_products[df_products['asin'] == asin]
                if not prod_row.empty:
                    title = str(prod_row[self.title_col].iloc[0])
                    texts = [title] + product_reviews['text'].fillna('').astype(str).tolist()
                    if len(texts) > 1 and any([t.strip() for t in texts[1:]]):
                        tfidf = self.tfidf
                        tfidf_matrix = tfidf.fit_transform(texts)
                        title_vec = tfidf_matrix[0:1]
                        review_vecs = tfidf_matrix[1:]
                        if review_vecs.shape[0] > 0:
                            title_sim = float(cosine_similarity(title_vec, review_vecs).mean())
        except Exception:
            title_sim = 0.0

        # Weighted product trust
        p_trust = (
            0.35 * (avg_rating / 5.0) +
            0.20 * rn_conf +
            0.15 * v_share +
            0.15 * price_factor +
            0.15 * title_sim
        )
        return np.clip(p_trust, 0, 1)

    def seller_trust_score(self, df_reviews, df_products, asin):
        """Calculate seller trustworthiness score"""
        try:
            if not self.seller_col or self.seller_col not in df_products.columns:
                return 0.5
            prod_row = df_products[df_products['asin'] == asin]
            if prod_row.empty:
                return 0.5
            seller_id = prod_row[self.seller_col].iloc[0]
            seller_products = df_products[df_products[self.seller_col] == seller_id]['asin'].unique()
            if len(seller_products) == 0:
                return 0.5
            scores = []
            for a in seller_products:
                try:
                    scores.append(self.product_trust_score(df_reviews, df_products, a))
                except Exception:
                    continue
            if len(scores) == 0:
                return 0.5
            return float(np.nanmean(scores))
        except Exception:
            return 0.5
    
    def final_product_score(self, df_reviews, df_products, asin, user_id=None):
        """Calculate final combined trust score"""
        p_trust = self.product_trust_score(df_reviews, df_products, asin)
        s_trust = self.seller_trust_score(df_reviews, df_products, asin)
        u_trust = self.user_trust_score(df_reviews, user_id) if user_id else 0.5
        
        final_score = 0.55 * p_trust + 0.35 * u_trust + 0.10 * s_trust
        return {
            'asin': asin, 'product_trust': round(p_trust, 3),
            'user_trust': round(u_trust, 3), 'seller_trust': round(s_trust, 3),
            'final_trust_score': round(final_score, 3)
        }
