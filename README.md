# Pathology WSI Deep Learning Pipeline

This project provides a pipeline for pathology whole slide image (WSI) analysis using deep learning, including patch data generation, model inference, and evaluation.

## 1. Quick Setup

Install all required packages and system libraries:

```bash
bash AutoSetup_pytorch.sh
```

## 2. Key Directory Structure

```
.
├── training_controller.py        # 控制整個訓練與測試流程
├── config.py                     # 全域參數設定
├── train_list.txt / val_list.txt / test_list.txt / unlabel_list.txt
├── PROCESSED_DATA/
│   └── CASE_UUID/
├── liver/
│   ├── masks/
│   └── tifs/
├── PRE_PRO/
│   ├── preprocess_workflow.py    # 前處理主控腳本
│   ├── save_case_pkl.py          # 生成 patch pkl 檔
│   ├── Gen_mask.py               # 將 XML 標註轉換成tiff
│   ├── json2xml.py               # 標註格式轉換
│   └── config.py                 # 前處理參數
├── TRAIN/
│   ├── main_execution.py         # 單一訓練階段主程式
│   ├── dataloader.py             # 訓練資料讀取
│   ├── aug_dataloader.py         # 增強資料讀取
│   ├── model.py                  # 模型架構
│   └── transform.py              # 資料增強與權重調整
├── TEST/
│   ├── pseudo_controller.py      # 產生 pseudo label 與測試
│   ├── Back_out.py               # 特殊測試/回退
│   ├── test.py                   # 批次推論主程式
│   ├── config_test.py            # 推論參數設定
│   ├── model.py                  # 推論用模型架構
│   ├── evaluation.py             # 評估指標計算
│   └── dataloader_pyvips.py      # 測試資料讀取
```

## 3. Dataset Organization

The project uses a hierarchical data organization:

- **Training set**: 15 samples → `train_list.txt`
- **Validation set**: 6 samples → `val_list.txt`
- **Test set**: 34 samples → `test_list.txt`
- **Unlabeled set**: 70 samples → `unlabel_list.txt`

Data splits are defined in `case_id.txt` with group assignments.

## 4. Model Architecture

The pipeline uses a custom dual-branch CNN architecture:

- **High-resolution branch (20x)**: Captures fine-grained details
- **Low-resolution branch (5x)**: Captures global context with ASPP
- **Multi-scale fusion**: Combines features from both branches
- **SE attention**: Optional channel attention mechanism

## 5. Basic Usage

### 5.1 Training Pipeline

**Automated training (recommended):**
```bash
python training_controller.py
```
This runs multiple training rounds with different iteration counts: `[5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 26000, 33000]`

**Manual single training:**
```bash
cd TRAIN
python main_execution.py
```

### 5.2 Data Preprocessing

1. **Place raw data:**
   - WSI files in `liver/tifs/`
   - Annotation masks in `liver/masks/`

2. **Run preprocessing:**
   ```bash
   cd PRE_PRO
   python preprocess_workflow.py
   ```

### 5.3 Inference and Evaluation

- **Edit configuration:**
  - Set paths and parameters in `config.py` and `TEST/config_test.py` as needed.
- **Prepare model weights:**
  - Place your trained model weights in the correct location and update the config.
- **Run inference:**
  - Go to the `TEST/` directory and run:
    ```bash
    python test.py
    ```
  - Results will be saved to `result.csv`.

## 6. Configuration Files

### 6.1 Global Config - `config.py`
```python
# Key parameters
patch_size = 256              # Patch size for training
stride_size = 128             # Stride for patch extraction
level = 0                     # WSI reading level
train_batch_size = 40         # Training batch size
```

### 6.2 Test Config - `TEST/config_test.py`
```python
# Inference parameters
test_batch_size = 64          # Test batch size
num_class = 2                 # Number of classes
threshold = [0.65, 0.6, 0.55, 0.5]  # Confidence thresholds
```

## 7. Output and Results

- **Training logs**: Saved in `TRAIN/LOG/`
- **Model weights**: Best model saved as `best_model.pt`
- **Inference results**: CSV file with IOU, F1 scores, and processing time
- **Evaluation metrics**: Computed automatically during inference

## 8. Requirements

- Python 3.8+
- PyTorch with CUDA support
- OpenSlide and pyvips for WSI handling
- See `AutoSetup_pytorch.sh` for complete dependency list

## 9. Troubleshooting

- **CUDA errors**: Check GPU drivers and CUDA version compatibility
- **Memory issues**: Reduce batch size or patch size
- **File reading errors**: Verify file formats and paths
- **Model loading errors**: Check model weight file integrity

For detailed logs, check:
- Training: `TRAIN/wandb/` and `TRAIN/LOG/`
- Errors: `TEST/patch_error/` and `TEST/excel_error/`
- Timing: `time.txt`