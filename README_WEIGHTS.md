# Model Weights & University Submission Notice

**Candidate**: Mahesh Kadam (Roll Number: 51, PRN: 202401120074)  
**Department**: Department of Data Science  
**Institution**: MIT Academy of Engineering (MITAOE)  

### Why is `model.safetensors` excluded from this zip package?
The trained weights file `model.safetensors` is 267.8 MB uncompressed. Most university submission portals (e.g. Moodle, Google Classroom, ERP) enforce a strict **10 MB upload limit**.

To comply with the 10 MB submission limit, this zip package excludes the 267 MB binary tensor file, bringing the package size down to **1.5 MB** while preserving 100% of the code, reports, presentation, datasets, metrics, and web dashboard.

### How to reproduce the exact weights:
Run the fine-tuning script on any machine (takes only ~88 seconds on Apple Silicon MPS or 1-2 minutes on CUDA/CPU):
```bash
python3 train.py
```
This will automatically generate and save `model.safetensors` directly into this folder.
