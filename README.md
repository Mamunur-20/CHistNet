# CHistNet: Histopathology Image Classification Using Supervised Contrastive Deep Learning

This repository contains the implementation of **CHistNet**, a deep learning framework designed for histopathology image classification using supervised contrastive learning. This code is associated with the following publication:

**"Histopathology Image Classification Using Supervised Contrastive Deep Learning"**  
Md Mamunur Rahaman, Ewan K. A. Millar, Erik Meijering  
University of New South Wales, Sydney, Australia, NSW Health Pathology, St. George Hospital, Kogarah, Australia.

---

## Abstract

In this study, we introduce **CHistNet**, a novel deep learning framework that leverages **supervised contrastive learning** alongside **cross-entropy loss** to enhance classification performance on histopathology images. The model is first pretrained using contrastive loss to learn robust image representations, followed by fine-tuning with cross-entropy loss for the classification task. This approach achieves state-of-the-art results across multiple histopathology image datasets.

---

## Datasets

The following datasets were used for training and evaluation:

- **BRACS**
- **BACH**
- **HE-GHI-DS**
- **MHIST**
- **EBHI**

Please refer to the paper for detailed instructions on how to download and prepare these datasets.

---
## How to Run

### 1. Generate TFRecords

Before training, the datasets must be converted into TensorFlow Records (TFRecords) to optimize the input pipeline. You can generate the TFRecords by running the `generate_tfrecords.py` script. Make sure to update the file paths within the script to match your local dataset directories.


01. python generate_tfrecords.py
02. python main.py

---
## Citation
Please cite the following paper:
```bash
@inproceedings{rahaman2024histopathology,
  title={Histopathology Image Classification Using Supervised Contrastive Deep Learning},
  author={Rahaman, Md Mamunur and Millar, Ewan K. A. and Meijering, Erik},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2024},
  organization={IEEE}
}


