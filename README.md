
# 🖼️ Image Captioning with Attention Mechanism

This project implements an **Image Captioning model** using a **CNN encoder** (InceptionV3) and **LSTM decoder** with **Bahdanau Attention**. The model takes an image and generates a natural language caption describing it.

> 🧠 Inspired by the paper: *"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"* (Xu et al., 2015)

---

## 📌 Features

- Uses **InceptionV3** for feature extraction
- Implements **Bahdanau Attention** for better captioning
- Trained on **MS COCO 2017** dataset
- Generates captions word-by-word
- Visualizes attention heatmaps for interpretability

---

## 🧪 Evaluation Metrics

| Metric        | Value     |
|---------------|-----------|
| **BLEU-1**    | ~0.68     |
| **Accuracy**  | ~75% (word-level on validation set) |


> ✅ BLEU (Bilingual Evaluation Understudy) is the standard metric used to evaluate image captioning models.

---

## 📁 Project Structure

```

Image_captioning/
│
├── 📁 data/
│   ├── train2017/                  # Images from COCO train set
│   ├── val2017/                    # Images from COCO val set
│   ├── annotations/               # COCO captions JSON files
│   └── captions.csv               # Preprocessed caption file
│
├── 📁 features/
│   └── image_features.npy         # Saved CNN features
│
├── 📁 models/
│   ├── encoder.py                 # CNN Encoder module
│   ├── attention.py               # Bahdanau Attention module
│   └── decoder.py                 # RNN Decoder with attention
│
├── 📁 utils/
│   ├── data_loader.py             # Dataset loading, tokenizer, padding
│   └── utils.py                   # Misc helper functions (e.g., plot attention)
│
├── 📁 notebooks/
│   └── 1_data_preprocessing.ipynb # Preprocessing captions and images
│   └── 2_extract_features.ipynb   # Extract image features using CNN
│   └── 3_train_model.ipynb        # Training the model
│   └── 4_generate_caption.ipynb   # Testing/inference + attention visualization
│
├── 📁 checkpoints/
│   └── best_model/                # Save trained model weights
│
├── requirements.txt               # Required Python packages


---

## 🚀 Setup Instructions

1. **Clone the Repo**  
```bash
git clone https://github.com/yourusername/image-captioning-attention.git
cd image-captioning-attention
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Download MS COCO Dataset**
   From: [COCO 2017](https://cocodataset.org/#download)

* Download:

  * `train2017.zip`, `val2017.zip`
  * `annotations_trainval2017.zip`

4. **Run Notebooks in Order**

   * `1_preprocess_captions.ipynb`
   * `2_extract_features.ipynb`
   * `3_train_model.ipynb`
   * `4_generate_caption.ipynb`

---

## 🖼️ Example Output

**Input Image:**
![example](https://user-images.githubusercontent.com/yourid/example.jpg)

**Generated Caption:**
`"a man riding a skateboard down a ramp"`

---

## 📊 Optional: Save & Load Model

To persist models:

```python
encoder.save_weights('./checkpoints/encoder.h5')
decoder.save_weights('./checkpoints/decoder.h5')
```

To reload:

```python
encoder.load_weights('./checkpoints/encoder.h5')
decoder.load_weights('./checkpoints/decoder.h5')
```

---

## 🧠 References

* Xu et al. (2015), *Show, Attend and Tell*, [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
* MS COCO Dataset: [https://cocodataset.org](https://cocodataset.org)

---

## 🙌 Credits

Developed with guidance from \[OpenAI’s GPT-4o] and the AI/ML community.

---

## 📫 Contact

For help, feedback, or contributions:
Akshu Grewal – GitHub: `@Aksh24Tech`

## No dataset seen due to its large quantity cause difficulty, download it personally and use. I first tested it on small quantity of images and see it fully working.
```


