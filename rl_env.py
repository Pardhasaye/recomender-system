
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    RL_AVAILABLE = True
except ImportError:
    gym = object  # Fallback
    spaces = None
    RL_AVAILABLE = False

class TrustworthyRLEnvironment(gym.Env if RL_AVAILABLE else object):
    """
    MDP Formulation for Trustworthy Recommendations:
    
    States (S): [trust_score, rating_norm, verified_ratio, review_confidence, text_quality]
    Actions (A): Recommend threshold (0=safe, 1=risky)  
    Rewards (R): Trust accuracy + verified bonus + rating alignment
    Transitions (P): Next product state
    Discount (γ): 0.99 (long-term trust)
    """
    
    def __init__(self, df_reviews, df_products, trust_model):
        if not RL_AVAILABLE:
            raise ImportError("gymnasium not installed")
            
        super().__init__()
        
        self.df_reviews = df_reviews.reset_index(drop=True)
        self.df_products = df_products
        self.trust_model = trust_model
        self.n_products = df_reviews['asin'].nunique()
        self.product_asins = df_reviews['asin'].unique()
        self.current_product_idx = 0
        
        # STATE SPACE (5D): Trust features [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # ACTION SPACE: Trust threshold to recommend
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Precompute features for speed
        self._precompute_features()
    
    def _precompute_features(self):
        """MDP State Engineering"""
        features = []
        
        print("[INFO] Precomputing MDP state features...")
        for idx, asin in enumerate(self.product_asins):
            if idx % 50 == 0:
                print(f"  Processing product {idx}/{len(self.product_asins)}...")
            
            # Get trust scores
            trust_data = self.trust_model.final_product_score(
                self.df_reviews, self.df_products, asin
            )
            
            reviews = self.df_reviews[self.df_reviews['asin'] == asin]
            
            state = np.array([
                trust_data['final_trust_score'],           # S1: Trust score
                reviews['rating'].mean() / 5.0,            # S2: Rating norm
                reviews['verified_purchase'].mean(),       # S3: Verified ratio
                1 - np.exp(-len(reviews) / 1000),          # S4: Review confidence
                np.clip(reviews['text'].str.len().mean() / 500, 0, 1)  # S5: Text quality
            ], dtype=np.float32)
            
            # Ensure all values are valid
            state = np.nan_to_num(state, nan=0.5, posinf=1.0, neginf=0.0)
            state = np.clip(state, 0.0, 1.0)
            
            features.append(state)
        
        self.product_features = np.array(features)
        print(f"[DONE] Precomputed {len(features)} product state features")

    def get_product_state(self, asin):
        """Get state vector for a specific product ASIN"""
        if asin not in self.product_asins:
            return None
        idx = np.where(self.product_asins == asin)[0][0]
        return self.product_features[idx]
    
    def reset(self, seed=None, options=None):
        """MDP: Reset to random state"""
        if seed is not None:
            np.random.seed(seed)
        self.current_product_idx = np.random.randint(0, self.n_products)
        obs = self.product_features[self.current_product_idx]
        return obs, {}
    
    def step(self, action):
        """MDP: Action → Reward → Next State"""
        threshold = float(action[0])  # Learned trust threshold
        
        # Current state
        idx = self.current_product_idx
        true_trust = float(self.product_features[idx, 0])
        true_rating = float(self.product_features[idx, 1])
        verified_ratio = float(self.product_features[idx, 2])
        
        # Decision
        recommend = true_trust > threshold
        
        # MDP REWARD FUNCTION (Trust-Optimized)
        trust_accuracy = 1 - abs(true_trust - threshold)           # Accuracy
        rating_reward = true_rating if recommend else 0.2          # Conservative
        verified_bonus = verified_ratio * 0.3                      # Trust signal
        
        reward = (0.5 * trust_accuracy + 
                 0.3 * rating_reward + 
                 0.2 * verified_bonus)
        
        # Next state (Markov property: independent)
        self.current_product_idx = np.random.randint(0, self.n_products)
        next_obs = self.product_features[self.current_product_idx]
        
        terminated = False
        truncated = False
        info = {
            'true_trust': true_trust,
            'recommended': recommend,
            'threshold': threshold,
            'reward_breakdown': {
                'trust_accuracy': trust_accuracy,
                'rating_reward': rating_reward,
                'verified_bonus': verified_bonus
            }
        }
        
        return next_obs, reward, terminated, truncated, info
