
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

# Module Imports
try:
    from config import REVIEWS_FILE, META_FILE, OUTPUT_DIR
    from preprocessing import clean_text
    from data_loader import load_jsonl
    from trust_model import TrustworthyRecommender
    from rl_env import TrustworthyRLEnvironment
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    # Fallback to local import if needed (e.g. if running as script vs package)
    pass

# RL Imports (optional — wrapped to allow running without RL dependencies)
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except Exception:
    RL_AVAILABLE = False
    PPO = None

warnings.filterwarnings('ignore')

def main():
    """Complete integrated pipeline with all functionalities"""
    print("[START] COMPLETE AMAZON REVIEWS TRUST PIPELINE + MDP RL")
    print("="*70)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store pipeline report
    report = {
        "pipeline_name": "Amazon Software Reviews Trust Pipeline + RL",
        "timestamp": pd.Timestamp.now().isoformat(),
        "steps": {},
        "final_statistics": {}
    }
    
    # ===== STEP 1: LOAD FILES =====
    print("\n[STEP 1] LOADING DATA")
    print("="*60)
    df_reviews = load_jsonl(REVIEWS_FILE)
    df_products = load_jsonl(META_FILE)
    
    report["steps"]["load_data"] = {
        "description": "Loading review and product metadata from JSONL files",
        "reviews_loaded": len(df_reviews),
        "products_loaded": len(df_products),
        "review_columns": list(df_reviews.columns) if not df_reviews.empty else []
    }
    
    if df_reviews.empty:
        print("\n[ERROR] NO REVIEWS! Check paths above ^")
        return
    
    # ===== STEP 2: STANDARDIZE COLUMNS =====
    print("\n[STEP 2] STANDARDIZING COLUMNS")
    print("="*60)
    review_cols = {
        'reviewerID': 'user_id', 'asin': 'asin', 'overall': 'rating', 
        'reviewText': 'text', 'unixReviewTime': 'timestamp', 
        'verified': 'verified_purchase', 'vote': 'helpful_vote'
    }
    df_reviews = df_reviews.rename(columns={k:v for k,v in review_cols.items() if k in df_reviews.columns})
    print(f"[DONE] Standardized {len(review_cols)} column mappings")
    
    # ===== STEP 3: FIX DATA TYPES =====
    print("\n[STEP 3] FIXING DATA TYPES")
    print("="*60)
    if 'rating' in df_reviews.columns:
        df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce').fillna(5.0)
    elif 'overall' in df_reviews.columns:
        df_reviews['rating'] = pd.to_numeric(df_reviews['overall'], errors='coerce').fillna(5.0)
    else:
        df_reviews['rating'] = 5.0
    if 'timestamp' in df_reviews.columns:
        df_reviews['timestamp'] = pd.to_numeric(df_reviews['timestamp'], errors='coerce')
    else:
        df_reviews['timestamp'] = 0
    if 'verified_purchase' in df_reviews.columns:
        df_reviews['verified_purchase'] = df_reviews['verified_purchase'].astype(bool)
    else:
        df_reviews['verified_purchase'] = False
    if 'helpful_vote' in df_reviews.columns:
        df_reviews['helpful_vote'] = pd.to_numeric(df_reviews['helpful_vote'], errors='coerce').fillna(0)
    elif 'vote' in df_reviews.columns:
        df_reviews['helpful_vote'] = pd.to_numeric(df_reviews['vote'], errors='coerce').fillna(0)
    else:
        df_reviews['helpful_vote'] = 0
    
    if len(df_products) > 0:
        df_products['price'] = pd.to_numeric(df_products.get('price', 0), errors='coerce').fillna(0)
    
    data_types = {
        "rating": "Converted to float64",
        "timestamp": "Converted to int64",
        "verified_purchase": "Converted to bool",
        "helpful_vote": "Converted to int64"
    }
    for dtype_info, desc in data_types.items():
        print(f"  * {desc}")
    
    # Determine dataset-level price mean and available product columns
    mean_price = None
    if len(df_products) > 0:
        prices = pd.to_numeric(df_products.get('price', pd.Series([], dtype=float)), errors='coerce')
        prices = prices.replace(0, np.nan).dropna()
        if len(prices) > 0:
            mean_price = float(prices.mean())
        else:
            mean_price = np.nan
    
    # Detect likely title and seller columns in product metadata
    title_col = None
    for c in ['title', 'productTitle', 'name', 'product_name']:
        if c in df_products.columns:
            title_col = c
            break
    seller_col = None
    for c in ['seller', 'brand', 'manufacturer', 'vendor']:
        if c in df_products.columns:
            seller_col = c
            break

    # Normalize ASIN / product id column in product metadata to 'asin'
    asin_like = next((c for c in ['asin', 'ASIN', 'product_id', 'id', 'parent_asin', 'sku'] if c in df_products.columns), None)
    if asin_like and asin_like != 'asin':
        df_products['asin'] = df_products[asin_like]
    elif 'asin' not in df_products.columns:
        df_products['asin'] = np.nan

    print(f"Detected - Product ID: {asin_like or 'none'}, Title: {title_col}, Seller: {seller_col}")
    
    # ===== STEP 4: TEXT CLEANING =====
    print("\n[STEP 4] TEXT CLEANING & PREPROCESSING")
    print("="*60)
    
    # Show before/after example
    sample_idx = 0
    sample_before = df_reviews['text'].iloc[sample_idx] if len(df_reviews) > 0 else ""
    df_reviews['clean_text'] = df_reviews['text'].fillna('').apply(clean_text)
    sample_after = df_reviews['clean_text'].iloc[sample_idx] if len(df_reviews) > 0 else ""
    
    report["steps"]["text_cleaning"] = {
        "description": "Text preprocessing for NLP models",
        "sample_before": sample_before[:150] if len(sample_before) > 0 else "N/A",
        "sample_after": sample_after[:150] if len(sample_after) > 0 else "N/A"
    }
    
    print(f"\n  Example (Cleaned):\n  {sample_after[:100]}...")
    
    df_reviews['text'] = df_reviews['clean_text']
    
    # ===== STEP 5: TRUST SCORING =====
    print("\n[STEP 5] COMPUTING TRUST SCORES")
    print("="*60)
    
    trust = TrustworthyRecommender()
    # Attach dataset metadata to trust object for use in scoring
    trust.mean_price = mean_price
    trust.title_col = title_col
    trust.seller_col = seller_col
    
    # Calculate trust scores for all products
    print(f"\n[INFO] Calculating trust scores for {df_reviews['asin'].nunique()} unique products...")
    product_trust_scores = {}
    
    unique_asins = df_reviews['asin'].unique()
    for idx, asin in enumerate(unique_asins[:12224]):  # Calculate for top 100 products
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{min(12224, len(unique_asins))} products processed...")
        
        product_reviews = df_reviews[df_reviews['asin'] == asin]
        if len(product_reviews) > 0:
            sample_user = product_reviews['user_id'].iloc[0]
            trust_data = trust.final_product_score(df_reviews, df_products, asin, sample_user)
            product_trust_scores[asin] = {
                "asin": asin,
                "product_trust": float(trust_data.get('product_trust', 0)),
                "user_trust": float(trust_data.get('user_trust', 0)) if pd.notna(trust_data.get('user_trust', np.nan)) else None,
                "seller_trust": float(trust_data.get('seller_trust', 0)) if pd.notna(trust_data.get('seller_trust', np.nan)) else None,
                "final_trust_score": float(trust_data.get('final_trust_score', 0)) if pd.notna(trust_data.get('final_trust_score', np.nan)) else None,
                "review_count": len(product_reviews),
                "avg_rating": float(product_reviews['rating'].mean()),
                "verified_ratio": float(product_reviews['verified_purchase'].mean())
            }
    
    print(f"[DONE] Calculated trust scores for {len(product_trust_scores)} products")
    
    # Show sample products
    sample_asins_list = list(product_trust_scores.items())[:3]
    for asin, score in sample_asins_list:
        print(f"\n  Product: {asin}")
        print(f"    Product Trust: {score['product_trust']:.3f}")
        print(f"    Final Score: {score['final_trust_score']}")
    
    # ===== STEP 6: PRODUCT ANALYSIS =====
    print("\n[STEP 6] PRODUCT ANALYSIS")
    print("="*60)
    
    # Sort products by price and display top 10 cheapest
    if not df_products.empty:
        price_col = next((col for col in ['price', 'Price', 'PRICE', 'price_usd', 'cost'] if col in df_products.columns), None)
        
        if price_col:
            # Create sorted dataframe by price
            sorted_products = df_products.copy()
            sorted_products[price_col] = pd.to_numeric(sorted_products[price_col], errors='coerce')
            sorted_products = sorted_products[sorted_products[price_col] > 0].sort_values(price_col)
            
            if not sorted_products.empty:
                print("\n[INFO] Top 10 Cheapest Products:")
                print("-" * 60)
                top10_cheap = sorted_products.head(10)
                for _, r in top10_cheap.iterrows():
                    title = r.get(title_col, r['asin']) if title_col and title_col in df_products.columns else r['asin']
                    if pd.isna(title): title = r['asin']
                    price = float(r[price_col]) if not pd.isna(r[price_col]) else 0.0
                    print(f"  • {str(title)[:50]}... - Cost: ${price:.2f}")

    # ===== STEP 7: SAVE OUTPUT FILES =====
    print("\n[STEP 7] SAVING OUTPUT FILES")
    print("="*60)
    output_file = Path(OUTPUT_DIR) / "trustworthy_software_reviews.parquet"
    df_reviews.to_parquet(output_file)
    print(f"[DONE] Parquet saved: {output_file}")
    
    # ===== STEP 8: MDP RL TRAINING (optional) =====
    if RL_AVAILABLE:
        print("\n[STEP 8] MDP RL TRAINING")
        print("="*60)

        # Create MDP Environment
        print("\n[INFO] Creating RL Environment...")
        rl_env = TrustworthyRLEnvironment(df_reviews, df_products, trust)

        # Train PPO
        print("\n[INFO] Training RL Policy with PPO...")
        rl_model = PPO(
            "MlpPolicy",
            rl_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )

        print("Training for 25,000 timesteps...")
        rl_model.learn(total_timesteps=25000)

        # Save RL Policy
        rl_policy_file = Path(OUTPUT_DIR) / "trust_rl_policy.zip"
        rl_model.save(str(rl_policy_file))
        print(f"\n[DONE] RL Policy saved: {rl_policy_file}")

        # ===== STEP 9: RL EVALUATION =====
        print("\n[STEP 9] RL EVALUATION")
        print("="*60)
        print("Evaluating learned policy on MDP environment...")

        obs, _ = rl_env.reset()
        total_reward = 0
        n_eval = 100
        
        for i in range(n_eval):
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, _, _, info = rl_env.step(action)
            total_reward += reward

        avg_reward = total_reward / n_eval
        print(f"  • Average MDP Reward: {avg_reward:.3f}")

        # ===== STEP 10: REAL-TIME PREDICTION DEMO =====
        print("\n[STEP 10] REAL-TIME PRODUCT PREDICTIONS")
        print("="*60)
        
        def predict_product_trust(asin_to_check):
            """Predict trust for a specific product using trained RL policy"""
            state = rl_env.get_product_state(asin_to_check)
            if state is None:
                print(f"[ERROR] Product {asin_to_check} not found in processed data.")
                return

            # Get RL Action (Threshold)
            action, _ = rl_model.predict(state, deterministic=True)
            threshold = float(action[0])
            
            # Re-calculate trust score components for display
            trust_details = trust.final_product_score(df_reviews, df_products, asin_to_check)
            product_trust_score = trust_details['final_trust_score']
            
            # Decision Logic
            is_trustworthy = product_trust_score > threshold
            decision_emoji = "[TRUSTWORTHY]" if is_trustworthy else "[RISKY]"
            
            print(f"\n[PREDICTION] PRODUCT: {asin_to_check}")
            print(f"   • Trust Score: {product_trust_score:.3f}")
            print(f"   • Threshold: {threshold:.3f}")
            print(f"   -> RECOMMENDATION: {decision_emoji}")

        # Demo with 3 random products
        print("Demonstrating prediction on 3 random products from dataset:")
        sample_asins = np.random.choice(rl_env.product_asins, 3, replace=False)
        for asin in sample_asins:
            predict_product_trust(asin)
            
    else:
        print("\n[WARN] RL libraries not available; skipping RL training and evaluation.")

    print("\n" + "="*70)
    print("[SUCCESS] COMPLETE PIPELINE + RL TRAINING SUCCESSFUL!")
    print("="*70)

if __name__ == "__main__":
    main()
