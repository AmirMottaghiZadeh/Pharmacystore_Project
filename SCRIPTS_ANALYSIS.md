# Scripts Analysis

## Core Scripts (Keep)
- `run_pipeline.py` - Entry point for pipeline
- `train_model.py` - Entry point for training
- `run_all.py` - Full workflow orchestrator
- `build_holiday_summary.py` - Data preparation utility
- `restore_db.sh` - Database restoration

## Utility Scripts (Keep - useful for debugging)
- `classify_atc.py` - ATC classification preview
- `scale_sales.py` - Sales scaling preview
- `feature_preview.py` - Feature engineering preview
- `ablation_study.py` - Model ablation experiments

## Duplicate/Scratch Scripts (Remove)
- `experiment_scratch.py` - Duplicate of sort_by_prescription with different sort key
- `sort_by_prescription.py` - Scratch script, same functionality as experiment_scratch

**Recommendation**: Remove `experiment_scratch.py` and `sort_by_prescription.py` as they are scratch/debug scripts with duplicate functionality.
