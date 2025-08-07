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

## 3. Basic Usage

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
