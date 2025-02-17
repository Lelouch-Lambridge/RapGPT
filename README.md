# **RapGPT - AI-Powered Rap Lyrics Generator**

## **📌 Project Overview**

RapGPT is a fine-tuned **StableLM** model designed to generate rap lyrics in the style of various artists. The project scrapes lyrics from **Genius**, tokenizes the data, fine-tunes the model using **LoRA (Low-Rank Adaptation)**, and generates rap lyrics based on prompts.

## **📁 Project Structure**

```
RapGPT/
│── logs/
│   ├── error.log  # Error logs
│   ├── info.log   # Information logs
│
│── models/
│   ├── stablelm_base/  # Pretrained model (downloaded)
│   ├── fine_tuned_artist/  # Fine-tuned models for specific artists
│
│── scripts/
│   ├── download_model.py  # Downloads StableLM
│   ├── prep_dataset.py  # Tokenizes and prepares dataset
│   ├── fine_tune.py  # Fine-tunes StableLM with artist data
│   ├── lyrics_gen.py  # Generates rap lyrics from the trained model
│   ├── genius_scraper.py  # Scrapes lyrics from Genius.com and saves them
│
│── lyrics.db  # Database storing raw scraped lyrics
│── tokenized_lyrics.db  # Database storing tokenized lyrics
│── output/  # Generated lyrics output folder
│── requirements.txt  # Python dependencies
│── Makefile  # Automates tasks
│── README.md  # Project documentation
```

## **🚀 Setup Instructions**

### **1️⃣ Install Dependencies**

Ensure Python 3.8+ is installed, then run:

```bash
pip install -r requirements.txt
```

### **2️⃣ Download StableLM Model**

Before fine-tuning, download the base model:

```bash
python scripts/download_model.py
```

This will save StableLM to `models/stablelm_base/`.

### **3️⃣ Scrape Rap Lyrics**

To scrape lyrics from Genius:

```bash
python scripts/genius_scraper.py
```

This stores the lyrics in `lyrics.db`.

### **4️⃣ Prepare Dataset**

Tokenize the lyrics before training:

```bash
python scripts/prep_dataset.py
```

This saves the tokenized dataset to `tokenized_lyrics.db`.

### **5️⃣ Fine-Tune the Model**

Fine-tune StableLM using an artist's lyrics:

```bash
python scripts/fine_tune.py
```

The fine-tuned model will be saved under `models/fine_tuned_artist/`.

### **6️⃣ Generate Lyrics**

To generate new rap lyrics based on a prompt:

```bash
python scripts/lyrics_gen.py
```

Lyrics will be saved in `output/generated_lyrics.txt`.

## **🛠 Using the Makefile (Optional)**

For convenience, you can automate tasks using `Makefile`:

```bash
make download_model   # Download StableLM
make scrape_lyrics    # Scrape lyrics
make prep_data        # Tokenize lyrics
make fine_tune        # Fine-tune model
make generate_lyrics  # Generate lyrics
```

## **🎯 Customization**

- **Fine-tune multiple artists** by modifying `fine_tune.py`.
- **Adjust lyric generation parameters** (temperature, top-k, top-p) in `lyrics_gen.py`.
- **Deploy a web app** using `Gradio` or `Streamlit`.

**Enjoy generating AI-powered rap lyrics! 🎤🔥**
