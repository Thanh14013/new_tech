"""
Example Usage - Production Pipeline
Demonstrates different ways to use the D60 LTV prediction tool
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_total.production_pipeline import ProductionPipeline
import pandas as pd


def example_1_single_prediction():
    """Example 1: Single campaign prediction"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Campaign Prediction")
    print("="*80)
    
    pipeline = ProductionPipeline()
    
    result = pipeline.predict(
        app_id='com.example.game',
        campaign='summer_sale_2024',
        data={
            'installs': 5000,
            'cost': 2500,
            'revenue': 1200,
            'retention_d7': 0.35,
            'retention_d30': 0.15
        }
    )
    
    print(f"\n✅ Prediction:")
    print(f"   App:          {result['app_id']}")
    print(f"   Campaign:     {result['campaign']}")
    print(f"   D60 LTV:      ${result['predicted_d60_ltv']:.2f}")
    print(f"   Method:       {result['method']}")
    print(f"   Confidence:   {result['confidence']*100:.0f}%")


def example_2_batch_prediction():
    """Example 2: Batch prediction from CSV"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'app_id': ['com.game.app1', 'com.game.app2', 'com.game.app3'],
        'campaign': ['campaign_A', 'campaign_B', 'campaign_C'],
        'installs': [5000, 10000, 2000],
        'cost': [2500, 5000, 1000],
        'revenue': [1200, 2500, 400]
    })
    
    # Save to temp CSV
    temp_file = 'results/temp_batch_example.csv'
    Path('results').mkdir(exist_ok=True)
    sample_data.to_csv(temp_file, index=False)
    print(f"\n📝 Created sample data: {temp_file}")
    
    # Predict
    pipeline = ProductionPipeline()
    results = pipeline.predict_batch(
        input_data=temp_file,
        output_path='results/temp_batch_predictions.csv'
    )
    
    print(f"\n📊 Results:")
    print(results[['app_id', 'campaign', 'predicted_d60_ltv', 'method', 'confidence']])


def example_3_validation_set():
    """Example 3: Evaluate on full validation set"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Validation Set Evaluation")
    print("="*80)
    
    val_path = 'data/features/validation_enhanced.csv'
    
    if not Path(val_path).exists():
        print(f"⚠️  Validation file not found: {val_path}")
        print(f"   Run steps 1-6 first to generate validation data")
        return
    
    pipeline = ProductionPipeline()
    results = pipeline.evaluate(val_path)
    
    print(f"\n✅ Evaluated {len(results):,} campaigns")


def example_4_programmatic_api():
    """Example 4: Using as Python API"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Programmatic API Usage")
    print("="*80)
    
    # Initialize once
    pipeline = ProductionPipeline()
    
    # Predict multiple campaigns
    campaigns = [
        ('com.game.app', 'campaign_1', {'installs': 1000, 'cost': 500, 'revenue': 200}),
        ('com.game.app', 'campaign_2', {'installs': 2000, 'cost': 1000, 'revenue': 400}),
        ('com.game.app', 'campaign_3', {'installs': 500, 'cost': 250, 'revenue': 100}),
    ]
    
    print(f"\n🔮 Predicting {len(campaigns)} campaigns...")
    
    for app_id, campaign, data in campaigns:
        result = pipeline.predict(app_id, campaign, data)
        
        # Calculate ROI
        predicted_revenue = result['predicted_d60_ltv'] * data['installs']
        roi = ((predicted_revenue - data['cost']) / data['cost'] * 100) if data['cost'] > 0 else 0
        
        print(f"\n  {campaign:20s} → D60 LTV: ${result['predicted_d60_ltv']:7.2f}  ROI: {roi:+6.1f}%")


def example_5_convenience_function():
    """Example 5: Using convenience function"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Quick Convenience Function")
    print("="*80)
    
    from tool_total.production_pipeline import predict_campaign
    
    result = predict_campaign(
        app_id='com.game.app',
        campaign='quick_test',
        installs=1000,
        cost=500,
        revenue=200
    )
    
    print(f"\n✅ Quick prediction:")
    print(f"   D60 LTV: ${result['predicted_d60_ltv']:.2f}")
    print(f"   Method:  {result['method']}")


if __name__ == "__main__":
    print("="*80)
    print("D60 LTV PREDICTION TOOL - USAGE EXAMPLES")
    print("="*80)
    
    try:
        example_1_single_prediction()
        example_2_batch_prediction()
        example_3_validation_set()
        example_4_programmatic_api()
        example_5_convenience_function()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
