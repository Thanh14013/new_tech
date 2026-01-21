"""
CLI Interface for D60 LTV Prediction Tool

Usage:
    python tool_total/cli.py predict --app-id com.game.app --campaign summer2024 --installs 1000 --cost 500 --revenue 200
    python tool_total/cli.py batch --input campaigns.csv --output predictions.csv
    python tool_total/cli.py evaluate
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_total.production_pipeline import ProductionPipeline


def predict_single(args):
    """Handle single prediction command"""
    print("="*80)
    print("SINGLE CAMPAIGN PREDICTION")
    print("="*80)
    
    # Initialize pipeline
    pipeline = ProductionPipeline()
    
    # Prepare data
    data = {
        'installs': args.installs,
        'cost': args.cost,
        'revenue': args.revenue
    }
    
    # Add optional parameters
    if args.retention_d7:
        data['retention_d7'] = args.retention_d7
    if args.retention_d30:
        data['retention_d30'] = args.retention_d30
    
    # Predict
    result = pipeline.predict(
        app_id=args.app_id,
        campaign=args.campaign,
        data=data
    )
    
    # Display result
    print(f"\n{'='*80}")
    print("PREDICTION RESULT")
    print("="*80)
    print(f"  📱 App ID:             {result['app_id']}")
    print(f"  📢 Campaign:           {result['campaign']}")
    print(f"  💰 Predicted D60 LTV:  ${result['predicted_d60_ltv']:.2f}")
    print(f"  🔧 Method:             {result['method']}")
    print(f"  ✅ Confidence:         {result['confidence']*100:.0f}%")
    print(f"  📈 Multiplier:         {result['multiplier']:.2f}x")
    print(f"  💵 Base Revenue:       ${result['base_revenue']:.2f}")
    print("="*80)
    
    # ROI Analysis
    predicted_revenue = result['predicted_d60_ltv'] * args.installs
    roi = ((predicted_revenue - args.cost) / args.cost * 100) if args.cost > 0 else 0
    
    print(f"\n📊 CAMPAIGN ANALYSIS:")
    print(f"  Total Investment:      ${args.cost:,.2f}")
    print(f"  Expected D60 Revenue:  ${predicted_revenue:,.2f}")
    print(f"  Expected ROI:          {roi:+.1f}%")
    
    if roi > 50:
        print(f"  💚 Assessment:         EXCELLENT - High profitability")
    elif roi > 0:
        print(f"  💛 Assessment:         GOOD - Positive return")
    else:
        print(f"  ❌ Assessment:         POOR - Negative return")
    
    print("="*80)


def predict_batch(args):
    """Handle batch prediction command"""
    print("="*80)
    print("BATCH PREDICTION")
    print("="*80)
    
    # Check input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = ProductionPipeline()
    
    # Predict
    results = pipeline.predict_batch(
        input_data=args.input,
        output_path=args.output
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH PREDICTION SUMMARY")
    print("="*80)
    print(f"  Total campaigns:       {len(results):,}")
    print(f"  Avg predicted LTV:     ${results['predicted_d60_ltv'].mean():.2f}")
    print(f"  Min predicted LTV:     ${results['predicted_d60_ltv'].min():.2f}")
    print(f"  Max predicted LTV:     ${results['predicted_d60_ltv'].max():.2f}")
    
    # Method breakdown
    print(f"\n  Methods used:")
    method_counts = results['method'].value_counts()
    for method, count in method_counts.items():
        pct = count / len(results) * 100
        print(f"    {method:25s} {count:6,} ({pct:5.1f}%)")
    
    print("="*80)


def evaluate(args):
    """Handle evaluate command"""
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Initialize pipeline
    pipeline = ProductionPipeline()
    
    # Evaluate
    results = pipeline.evaluate()
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results contain {len(results):,} predictions")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='D60 LTV Prediction Tool - Production Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python tool_total/cli.py predict \\
      --app-id com.game.app \\
      --campaign summer_sale_2024 \\
      --installs 5000 \\
      --cost 2500 \\
      --revenue 1200

  # Batch prediction
  python tool_total/cli.py batch \\
      --input data/campaigns.csv \\
      --output results/predictions.csv

  # Model evaluation
  python tool_total/cli.py evaluate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict D60 LTV for a single campaign')
    predict_parser.add_argument('--app-id', required=True, help='Application ID (e.g., com.game.app)')
    predict_parser.add_argument('--campaign', required=True, help='Campaign name')
    predict_parser.add_argument('--installs', type=int, required=True, help='Number of installs')
    predict_parser.add_argument('--cost', type=float, required=True, help='Campaign cost in USD')
    predict_parser.add_argument('--revenue', type=float, required=True, help='D1 revenue in USD')
    predict_parser.add_argument('--retention-d7', type=float, help='Optional: 7-day retention rate (0-1)')
    predict_parser.add_argument('--retention-d30', type=float, help='Optional: 30-day retention rate (0-1)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch predict from CSV file')
    batch_parser.add_argument('--input', required=True, help='Input CSV file path')
    batch_parser.add_argument('--output', required=True, help='Output CSV file path')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on validation set')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'predict':
        predict_single(args)
    elif args.command == 'batch':
        predict_batch(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
