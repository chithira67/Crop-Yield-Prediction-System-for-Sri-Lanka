import os
import sys

# Set UTF-8 encoding for Windows console compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    """Run the complete pipeline"""
    print("=" * 60)
    print("Crop Yield Prediction System - Complete Pipeline")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n[1/2] Processing Data...")
    print("-" * 60)
    try:
        from src.data_processing import process_data
        processed_df, crop_enc, district_enc, scaler_obj = process_data()
        print("[OK] Data processing completed successfully!")
    except Exception as e:
        print(f"[ERROR] Error in data processing: {str(e)}")
        return
    
    # Step 2: Model Training
    print("\n[2/2] Training Models...")
    print("-" * 60)
    try:
        from src.train_models import train_all_models
        results = train_all_models()
        print("[OK] Model training completed successfully!")
        print(f"\nBest Model: {results['best_model_name'].upper()}")
        print(f"Best RMSE: {results['metrics'][results['best_model_name']]['RMSE']:.4f}")
        print(f"Best R²: {results['metrics'][results['best_model_name']]['R²']:.4f}")
    except Exception as e:
        print(f"[ERROR] Error in model training: {str(e)}")
        return
    
    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'streamlit run src/app.py' to launch the dashboard")
    print("2. Open notebooks/ for detailed analysis")
    print("3. Check model/ directory for trained models")
    print("4. Check data/processed/ for processed features")


if __name__ == "__main__":
    main()

