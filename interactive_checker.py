
import numpy as np
from recommendation_display import display_product_recommendation, display_batch_recommendations


class InteractiveProductChecker:
    """Interactive CLI for checking product trustworthiness"""
    
    def __init__(self, rl_env, trained_model=None, default_threshold=0.5):
        """
        Initialize the interactive checker
        
        Args:
            rl_env: TrustworthyRLEnvironment instance
            trained_model: Optional trained RL model for predictions
            default_threshold: Default trust threshold if no model provided
        """
        self.rl_env = rl_env
        self.trained_model = trained_model
        self.default_threshold = default_threshold
    
    def get_rl_decision(self, state):
        """
        Get RL model decision for a given state
        
        Args:
            state: Product state vector
            
        Returns:
            tuple: (decision, threshold)
        """
        if self.trained_model is not None:
            try:
                # Use trained model to predict threshold
                action = self.trained_model.predict(state, deterministic=True)[0]
                threshold = float(action[0])
            except Exception:
                threshold = self.default_threshold
        else:
            threshold = self.default_threshold
        
        # Decision: recommend if trust score > threshold
        trust_score = float(state[0])  # First element is final trust score
        decision = trust_score > threshold
        
        return decision, threshold
    
    def check_specific_asin(self, asin):
        """
        Check trustworthiness of a specific product ASIN
        
        Args:
            asin: Product ASIN to check
            
        Returns:
            bool: True if successful, False if ASIN not found
        """
        # Get product state and trust data
        state = self.rl_env.get_product_state(asin)
        trust_data = self.rl_env.get_product_trust_data(asin)
        
        if state is None or trust_data is None:
            print(f"\n❌ Error: ASIN '{asin}' not found in dataset.")
            return False
        
        # Get RL decision
        decision, threshold = self.get_rl_decision(state)
        
        # Display detailed recommendation
        display_product_recommendation(asin, trust_data, decision, threshold)
        
        return True
    
    def check_random_products(self, n):
        """
        Check N random products from the dataset
        
        Args:
            n: Number of random products to check
            
        Returns:
            bool: True if successful
        """
        # Validate input
        max_products = len(self.rl_env.product_asins)
        if n > max_products:
            print(f"\n⚠️  Warning: Requested {n} products, but only {max_products} available.")
            n = max_products
        
        # Randomly select N products
        random_indices = np.random.choice(max_products, size=n, replace=False)
        random_asins = self.rl_env.product_asins[random_indices]
        
        # Collect results
        results = []
        for asin in random_asins:
            state = self.rl_env.get_product_state(asin)
            trust_data = self.rl_env.get_product_trust_data(asin)
            decision, threshold = self.get_rl_decision(state)
            
            results.append({
                'asin': asin,
                'trust_data': trust_data,
                'decision': decision,
                'threshold': threshold
            })
        
        # Display batch results
        display_batch_recommendations(results)
        
        return True
    
    def run(self):
        """Run the interactive CLI interface"""
        print("\n" + "="*70)
        print("TRUSTWORTHY PRODUCT RECOMMENDATION SYSTEM")
        print("="*70)
        print(f"\nDataset: {len(self.rl_env.product_asins)} products available")
        print(f"Default Threshold: {self.default_threshold:.3f}")
        
        while True:
            print("\n" + "-"*70)
            print("MAIN MENU:")
            print("  1. Check specific ASIN")
            print("  2. Check N random products")
            print("  3. Exit")
            print("-"*70)
            
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == '1':
                    # Check specific ASIN
                    asin = input("\nEnter product ASIN: ").strip()
                    if not asin:
                        print("❌ Error: ASIN cannot be empty.")
                        continue
                    self.check_specific_asin(asin)
                
                elif choice == '2':
                    # Check random products
                    n_input = input("\nEnter number of random products to check: ").strip()
                    try:
                        n = int(n_input)
                        if n <= 0:
                            print("❌ Error: Number must be positive.")
                            continue
                        self.check_random_products(n)
                    except ValueError:
                        print("❌ Error: Invalid number format.")
                        continue
                
                elif choice == '3':
                    # Exit
                    print("\n👋 Thank you for using the Trustworthy Product Recommendation System!")
                    print("="*70 + "\n")
                    break
                
                else:
                    print("❌ Error: Invalid choice. Please enter 1, 2, or 3.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Exiting...")
                print("="*70 + "\n")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue


def run_interactive_mode(rl_env, trained_model=None, default_threshold=0.5):
    """
    Convenience function to run interactive mode
    
    Args:
        rl_env: TrustworthyRLEnvironment instance
        trained_model: Optional trained RL model
        default_threshold: Default trust threshold
    """
    checker = InteractiveProductChecker(rl_env, trained_model, default_threshold)
    checker.run()
