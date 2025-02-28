# SDGNN: Structure-aware Dual Graph Neural Network forCode Summarization

[![Paper]](https://link.springer.com/article/10.1007/s13042-024-02471-2)

This repository contains the implementation of the Structure-aware Dual Graph Neural Network (SDGNN), as described in our paper: **SDGNN: Structure-aware Dual Graph Neural Network for Code Summarization**.

## Data Preparation
The default dataset is Conala, with data stored in the `data/` folder. For WikiSQL and ATIS datasets:
- Place WikiSQL data in `wikisql/` folder
- Place ATIS data in `atis/` folder
- Each dataset folder should contain corresponding node type and edge type files


## Setup

### 1. Modify Configuration
- **Configuration File:** `data_util/config.py`
  - **1.1. Data Paths:**  
    Update the data paths (e.g., `train_data_path`, `eval_data_path`, `decode_data_path`, `vocab_path`) to point to the correct directories for the dataset you are using.
  - **1.2. Model Parameters:**  
    Adjust the `max_layer` parameter based on the specific dataset.

### 2. Training
Run the following command to start training:
```bash
CUDA_VISIBLE_DEVICES=1 nohup bash start_train.sh > train.log 2>&1 &
```

### 3. Inference
Run the following command for inference:
```bash
CUDA_VISIBLE_DEVICES=0 bash start_decode.sh <model_path> > decode.log 2>&1 &
```
For example:
```bash
CUDA_VISIBLE_DEVICES=0 bash start_decode.sh './log/train_1719852705/model/' > decode.log 2>&1 &
```
Note:
 - If you specify a folder, the script will load model weights sequentially.
 - If you specify a particular model weight file, only that model will be used for inference.


### 4. Evaluation
After running inference, calculate the evaluation scores with:
```bash
python compute_sore.py <reference_directory> <inference_directory>
```
For example:
```bash
python compute_sore.py './log/train_1719852705/decode/decode_model_2530_1719855491/rouge_ref/' './log/train_1719852705/decode/decode_model_2530_1719855491/rouge_dec_dir/'
```

This script will compare the generated summaries against the reference summaries and output the evaluation scores.


## Citation
If you use this project or find it useful in your research, please cite our paper:
```bibtex
@article{hao2025sdgnn,
  title={SDGNN: Structure-aware Dual Graph Neural Network for Code Summarization},
  author={Hao, Zhifeng and Lin, Zonghao and Zhang, Shengqiang and Xu, Boyan and Cai, Ruichu},
  journal={International Journal of Machine Learning and Cybernetics},
  pages={1--17},
  year={2025},
  publisher={Springer}
}
```