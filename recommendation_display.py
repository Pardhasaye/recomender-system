
def display_product_recommendation(asin, trust_data, rl_decision, threshold):
    """
    Display detailed product recommendation with trust scores
    
    Args:
        asin: Product ASIN
        trust_data: Dictionary containing trust scores and details
        rl_decision: Boolean indicating if RL model recommends the product
        threshold: Trust threshold used by RL model
    """
    print("\n" + "="*70)
    print(f"PRODUCT RECOMMENDATION REPORT: {asin}")
    print("="*70)
    
    # Trust Score Breakdown
    print("\n📊 TRUST SCORE BREAKDOWN:")
    print(f"  ├─ Product Trust:  {trust_data['product_trust']:.3f} / 1.000")
    print(f"  ├─ User Trust:     {trust_data['user_trust']:.3f} / 1.000")
    print(f"  ├─ Seller Trust:   {trust_data['seller_trust']:.3f} / 1.000")
    print(f"  └─ Final Trust:    {trust_data['final_trust_score']:.3f} / 1.000")
    
    # Detailed Features (if available)
    if 'details' in trust_data:
        details = trust_data['details']
        print("\n🔍 DETAILED FEATURES:")
        print(f"  ├─ Average Rating:      {details['avg_rating_norm']:.3f} (normalized)")
        print(f"  ├─ Verified Ratio:      {details['verified_ratio']:.3f}")
        print(f"  ├─ Review Confidence:   {details['review_confidence']:.3f}")
        print(f"  ├─ Text Quality:        {details['text_quality']:.3f}")
        print(f"  ├─ Price Factor:        {details['price_factor']:.3f}")
        print(f"  └─ Title Similarity:    {details['title_similarity']:.3f}")
    
    # RL Model Decision
    print("\n🤖 RL MODEL DECISION:")
    print(f"  ├─ Trust Threshold:     {threshold:.3f}")
    print(f"  ├─ Final Trust Score:   {trust_data['final_trust_score']:.3f}")
    print(f"  └─ Recommendation:      {'PASS' if rl_decision else 'FAIL'} (Trust {'>' if rl_decision else '≤'} Threshold)")
    
    # Final Verdict
    print("\n✅ VERDICT:")
    if rl_decision:
        print(f"  ✓ TRUSTWORTHY - Product {asin} is recommended")
        print(f"    Reason: Trust score ({trust_data['final_trust_score']:.3f}) exceeds threshold ({threshold:.3f})")
    else:
        print(f"  ✗ RISKY - Product {asin} is NOT recommended")
        print(f"    Reason: Trust score ({trust_data['final_trust_score']:.3f}) below threshold ({threshold:.3f})")
    
    print("="*70 + "\n")


def display_batch_recommendations(results):
    """
    Display multiple product recommendations in a formatted table
    
    Args:
        results: List of dictionaries containing product recommendation data
    """
    if not results:
        print("\n⚠️  No results to display.")
        return
    
    print("\n" + "="*100)
    print("BATCH PRODUCT RECOMMENDATIONS")
    print("="*100)
    
    # Table header
    header = f"{'ASIN':<15} {'Product':<8} {'User':<8} {'Seller':<8} {'Final':<8} {'Threshold':<10} {'Verdict':<12}"
    print(f"\n{header}")
    print("-"*100)
    
    # Table rows
    for result in results:
        asin = result['asin']
        trust = result['trust_data']
        threshold = result['threshold']
        decision = result['decision']
        
        verdict = "✓ TRUSTWORTHY" if decision else "✗ RISKY"
        
        row = (f"{asin:<15} "
               f"{trust['product_trust']:<8.3f} "
               f"{trust['user_trust']:<8.3f} "
               f"{trust['seller_trust']:<8.3f} "
               f"{trust['final_trust_score']:<8.3f} "
               f"{threshold:<10.3f} "
               f"{verdict:<12}")
        print(row)
    
    print("-"*100)
    
    # Summary statistics
    trustworthy_count = sum(1 for r in results if r['decision'])
    risky_count = len(results) - trustworthy_count
    avg_trust = sum(r['trust_data']['final_trust_score'] for r in results) / len(results)
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total Products:      {len(results)}")
    print(f"  Trustworthy:         {trustworthy_count} ({trustworthy_count/len(results)*100:.1f}%)")
    print(f"  Risky:               {risky_count} ({risky_count/len(results)*100:.1f}%)")
    print(f"  Average Trust Score: {avg_trust:.3f}")
    print("="*100 + "\n")


def display_simple_recommendation(asin, trust_data, rl_decision, threshold):
    """
    Display a simplified single-line recommendation
    
    Args:
        asin: Product ASIN
        trust_data: Dictionary containing trust scores
        rl_decision: Boolean indicating if RL model recommends the product
        threshold: Trust threshold used by RL model
    """
    verdict = "✓ TRUSTWORTHY" if rl_decision else "✗ RISKY"
    print(f"{asin}: {verdict} (Trust: {trust_data['final_trust_score']:.3f}, Threshold: {threshold:.3f})")
