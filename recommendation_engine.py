
import numpy as np
from product_similarity import ProductSimilarity


class RecommendationEngine:
    """Generate trustworthy product recommendations based on similarity"""
    
    def __init__(self, df_reviews, df_products, rl_env):
        """
        Initialize recommendation engine
        
        Args:
            df_reviews: DataFrame of product reviews
            df_products: DataFrame of product metadata
            rl_env: TrustworthyRLEnvironment instance for trust scores
        """
        self.df_reviews = df_reviews
        self.df_products = df_products
        self.rl_env = rl_env
        self.similarity_calculator = ProductSimilarity(df_reviews, df_products)
    
    def get_similar_trustworthy_products(self, target_asin, min_trust=0.5, top_n=10, 
                                        collaborative_weight=0.6, content_weight=0.4):
        """
        Get similar trustworthy product recommendations
        
        Args:
            target_asin: Product ASIN to find similar products for
            min_trust: Minimum trust threshold (0.0 to 1.0)
            top_n: Number of recommendations to return
            collaborative_weight: Weight for collaborative filtering
            content_weight: Weight for content-based filtering
            
        Returns:
            List of recommendation dictionaries with:
                - asin: Product ASIN
                - similarity: Similarity score
                - trust_data: Trust score breakdown
                - final_score: Combined similarity × trust score
                - method_breakdown: Collaborative and content scores
        """
        # Validate target product exists
        if target_asin not in self.rl_env.product_asins:
            return []
        
        # Get similar products using hybrid similarity
        similar_products = self.similarity_calculator.calculate_hybrid_similarity(
            target_asin, 
            top_n=100,  # Get more candidates for filtering
            collaborative_weight=collaborative_weight,
            content_weight=content_weight
        )
        
        # Filter by trust and calculate final scores
        trustworthy_products = []
        
        for asin, similarity, method_breakdown in similar_products:
            # Get trust data
            trust_data = self.rl_env.get_product_trust_data(asin)
            
            if trust_data is None:
                continue
            
            # Filter by minimum trust threshold
            if trust_data['final_trust_score'] >= min_trust:
                # Calculate final score: similarity × trust
                final_score = similarity * trust_data['final_trust_score']
                
                trustworthy_products.append({
                    'asin': asin,
                    'similarity': similarity,
                    'trust_data': trust_data,
                    'final_score': final_score,
                    'method_breakdown': method_breakdown
                })
        
        # Rank by final score
        trustworthy_products.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Return top N
        return trustworthy_products[:top_n]
    
    def explain_recommendation(self, target_asin, recommended_asin, similarity_score, 
                              trust_data, method_breakdown):
        """
        Generate human-readable explanation for why a product is recommended
        
        Args:
            target_asin: Original product ASIN
            recommended_asin: Recommended product ASIN
            similarity_score: Overall similarity score
            trust_data: Trust score breakdown
            method_breakdown: Collaborative and content scores
            
        Returns:
            String explanation
        """
        explanation_parts = []
        
        # Similarity explanation
        collab_score = method_breakdown.get('collaborative', 0.0)
        content_score = method_breakdown.get('content', 0.0)
        
        explanation_parts.append(f"Similarity: {similarity_score:.1%}")
        
        if collab_score > 0:
            explanation_parts.append(
                f"  • {collab_score:.1%} based on user review patterns"
            )
        
        if content_score > 0:
            explanation_parts.append(
                f"  • {content_score:.1%} based on product features"
            )
        
        # Trust explanation
        explanation_parts.append(f"\nTrust Score: {trust_data['final_trust_score']:.3f}")
        explanation_parts.append(f"  • Product Trust: {trust_data['product_trust']:.3f}")
        explanation_parts.append(f"  • Seller Trust: {trust_data['seller_trust']:.3f}")
        
        if 'details' in trust_data:
            details = trust_data['details']
            explanation_parts.append(
                f"  • Verified Purchase Ratio: {details['verified_ratio']:.1%}"
            )
        
        return "\n".join(explanation_parts)
    
    def get_product_info(self, asin):
        """
        Get basic product information for display
        
        Args:
            asin: Product ASIN
            
        Returns:
            Dictionary with product info or None if not found
        """
        product = self.df_products[self.df_products['asin'] == asin]
        
        if product.empty:
            return None
        
        product = product.iloc[0]
        
        return {
            'asin': asin,
            'title': product.get('title', 'Unknown Product')[:80],
            'category': product.get('main_category', 'Unknown'),
            'price': product.get('price', 0.0)
        }
    
    def get_recommendations_with_info(self, target_asin, min_trust=0.5, top_n=10):
        """
        Get recommendations with full product information
        
        Args:
            target_asin: Product ASIN to find similar products for
            min_trust: Minimum trust threshold
            top_n: Number of recommendations to return
            
        Returns:
            Dictionary with:
                - target_product: Target product info
                - recommendations: List of recommendations with product info
        """
        # Get target product info
        target_info = self.get_product_info(target_asin)
        
        if target_info is None:
            return None
        
        # Get trust data for target
        target_trust = self.rl_env.get_product_trust_data(target_asin)
        
        # Get recommendations
        recommendations = self.get_similar_trustworthy_products(
            target_asin, min_trust, top_n
        )
        
        # Add product info to each recommendation
        for rec in recommendations:
            rec['product_info'] = self.get_product_info(rec['asin'])
            rec['explanation'] = self.explain_recommendation(
                target_asin,
                rec['asin'],
                rec['similarity'],
                rec['trust_data'],
                rec['method_breakdown']
            )
        
        return {
            'target_product': target_info,
            'target_trust': target_trust,
            'recommendations': recommendations,
            'total_found': len(recommendations)
        }
