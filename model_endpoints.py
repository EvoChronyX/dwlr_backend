"""
Training and Model Management Endpoints for High-Accuracy Ensemble
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json

from high_accuracy_predictor import HighAccuracyEnsemblePredictor

router = APIRouter(prefix="/model", tags=["model"])

class TrainingRequest(BaseModel):
    retrain: bool = True
    use_uploaded_data: bool = False
    dataset_id: Optional[str] = None

class PredictionBatchRequest(BaseModel):
    data: list
    return_confidence: bool = True

@router.post("/train")
async def train_ensemble_model(request: TrainingRequest):
    """
    Train or retrain the high-accuracy ensemble model
    """
    try:
        from main import get_ensemble_model, BASE_DIR
        
        # Determine data source
        if request.use_uploaded_data and request.dataset_id:
            data_path = BASE_DIR / "data" / f"{request.dataset_id}.csv"
            if not data_path.exists():
                raise HTTPException(status_code=404, detail="Dataset not found")
        else:
            data_path = BASE_DIR / "train_dataset.csv"
            if not data_path.exists():
                raise HTTPException(status_code=404, detail="Default training dataset not found")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Initialize or get ensemble model
        ensemble = get_ensemble_model()
        if ensemble is None:
            ensemble = HighAccuracyEnsemblePredictor()
        
        # Train the model
        print(f"🚀 Starting ensemble training with {len(df)} samples...")
        metrics = ensemble.train_ensemble(df)
        
        # Save the model
        model_dir = BASE_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "ensemble_model.pkl"
        ensemble.save_model(str(model_path))
        
        return {
            "status": "success",
            "message": "Ensemble model trained successfully",
            "metrics": metrics,
            "model_path": str(model_path),
            "training_samples": metrics.get('training_samples', 'N/A'),
            "validation_samples": metrics.get('validation_samples', 'N/A')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/predict/batch")
async def batch_predictions(request: PredictionBatchRequest):
    """
    Make batch predictions using the ensemble model
    """
    try:
        from main import get_ensemble_model
        
        ensemble = get_ensemble_model()
        if not ensemble or not ensemble.is_trained:
            raise HTTPException(status_code=400, detail="Ensemble model not trained")
        
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        if request.return_confidence:
            predictions, lower_bounds, upper_bounds = ensemble.predict_with_confidence(df)
            
            results = []
            for i, (pred, lower, upper) in enumerate(zip(predictions, lower_bounds, upper_bounds)):
                results.append({
                    "index": i,
                    "prediction": float(pred),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "confidence": 0.95
                })
        else:
            predictions = ensemble.predict(df)
            results = [{"index": i, "prediction": float(pred)} for i, pred in enumerate(predictions)]
        
        return {
            "status": "success",
            "model_type": "high_accuracy_ensemble",
            "predictions": results,
            "total_predictions": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/status")
async def model_status():
    """
    Get current model status and availability
    """
    try:
        from main import get_ensemble_model, BASE_DIR
        
        # Check ensemble model
        ensemble_status = "not_loaded"
        ensemble_metrics = {}
        model_path = BASE_DIR / "models" / "ensemble_model.pkl"
        
        try:
            ensemble = get_ensemble_model()
            if ensemble and ensemble.is_trained:
                ensemble_status = "trained_and_loaded"
                ensemble_metrics = ensemble.model_metrics
            elif model_path.exists():
                ensemble_status = "saved_but_not_loaded"
        except Exception:
            ensemble_status = "error"
        
        # Check training data
        train_data_path = BASE_DIR / "train_dataset.csv"
        training_data_available = train_data_path.exists()
        
        return {
            "ensemble_model": {
                "status": ensemble_status,
                "metrics": ensemble_metrics,
                "model_file_exists": model_path.exists()
            },
            "training_data": {
                "available": training_data_available,
                "path": str(train_data_path) if training_data_available else None
            },
            "capabilities": {
                "can_train": training_data_available,
                "can_predict": ensemble_status == "trained_and_loaded"
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@router.get("/features")
async def get_feature_info():
    """
    Get information about features used by the ensemble model
    """
    try:
        from main import get_ensemble_model
        
        ensemble = get_ensemble_model()
        if not ensemble or not ensemble.is_trained:
            # Return feature engineering info even if not trained
            from feature_engineering import AdvancedFeatureEngineer
            feature_engineer = AdvancedFeatureEngineer()
            
            return {
                "status": "model_not_trained",
                "base_features": feature_engineer.feature_cols,
                "target": feature_engineer.target_col,
                "feature_types": {
                    "original": "Temperature_C, Rainfall_mm, pH, Dissolved_Oxygen_mg_L",
                    "lag_features": "1, 3, 6, 12 month lags",
                    "rolling_features": "3, 6, 12 month rolling statistics",
                    "seasonal": "Sine/cosine seasonal encoding",
                    "interactions": "Temperature-pH, Rainfall-pH, etc.",
                    "polynomials": "Squared terms for non-linear relationships"
                }
            }
        
        model_info = ensemble.get_model_info()
        
        return {
            "status": "success",
            "total_features": model_info.get("n_features", 0),
            "feature_categories": model_info.get("feature_categories", {}),
            "feature_engineering": {
                "lag_features": "Historical values for temporal patterns",
                "rolling_statistics": "Moving averages and volatility measures",
                "seasonal_encoding": "Cyclical time representations",
                "interaction_terms": "Feature combinations for complex relationships",
                "polynomial_features": "Non-linear transformations"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature info retrieval failed: {str(e)}")

@router.delete("/reset")
async def reset_model():
    """
    Reset the ensemble model (useful for retraining)
    """
    try:
        from main import ensemble_model, BASE_DIR
        import main
        
        # Reset global model
        main.ensemble_model = None
        
        # Optionally remove saved model file
        model_path = BASE_DIR / "models" / "ensemble_model.pkl"
        if model_path.exists():
            model_path.unlink()
        
        return {
            "status": "success",
            "message": "Model reset successfully. Use /model/train to retrain."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reset failed: {str(e)}")