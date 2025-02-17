# **RapGPT Lyrics Generator - Fine-Tuning StableLM**

## **📌 Project Overview**

This project fine-tunes **StableLM** by Stability AI to generate rap lyrics in the style of different artists. The model is trained using **Low-Rank Adaptation (LoRA)** for efficient fine-tuning.

## **📁 Directory Structure**

```
rap_lyrics_generator/
│── logs/
│   ├── error.log/
│   ├── info.log/
│
│── models/
│   ├── stablelm_base/   # Pretrained model (downloaded)
│
│── scripts/
│   ├── download_model.py  # Downloads StableLM
│   ├── prep_dataset.py  # Loads and tokenizes lyrics
│   ├── fine_tune.py  # Fine-tunes StableLM
│   ├── lyrics_gen.py  # Generates lyrics
│   ├── genius_scraper.py  # Scrape Lyrics from the genius.com
│
│── datasets.db
│── logs/  # Training logs
│── output/  # Generated lyrics
│── requirements.txt
│── README.md
```

## **🚀 Setup Instructions**

### **1️⃣ Install Dependencies**

Run the following command to install required Python packages:

```bash
pip install -r requirements.txt
```

### **2️⃣ Download StableLM Model**

Before fine-tuning, download the base model:

```bash
python scripts/1_download_model.py
```

This will save StableLM to `models/stablelm_base/`.

### **3️⃣ Prepare Dataset**

Ensure rap lyrics for different artists are inside `datasets/`. Then, run:

```bash
python scripts/2_prepare_dataset.py
```

This will tokenize and save the dataset to `datasets/tokenized/`.

### **4️⃣ Fine-Tune the Model**

Fine-tune StableLM on an artist's lyrics (e.g., Tupac):

```bash
python scripts/3_finetune_stablelm.py
```

The fine-tuned model will be saved in `models/stablelm_rap_tupac/`.

### **5️⃣ Generate Lyrics**

To generate new rap lyrics:

```bash
python scripts/4_generate_lyrics.py
```

Lyrics will be saved in `output/generated_lyrics.txt`.

## **🎯 Customization**

- **Fine-tune multiple artists** by modifying `3_finetune_stablelm.py`.
- **Adjust lyric generation parameters** (temperature, top-k, top-p) in `4_generate_lyrics.py`.
- **Deploy a web app** using `Gradio` or `Streamlit`.

## **📌 Next Steps**

✅ Add multi-artist fine-tuning
✅ Create an interactive web app
✅ Deploy model via API

**Enjoy generating AI-powered rap lyrics! 🎤🔥**
