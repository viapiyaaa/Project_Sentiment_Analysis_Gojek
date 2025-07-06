# 🧠 Sentiment Analysis of Gojek App Reviews

This project analyzes user reviews of the **Gojek** mobile application from Google Play Store and classifies them into **positive**, **neutral**, or **negative** sentiments using NLP and deep learning techniques.

[![Streamlit App](https://img.shields.io/badge/-Open%20App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://sentiment-analysis-gojek.streamlit.app/)

---

## 📌 Features

- Scraping user reviews from Google Play Store using `google-play-scraper`.
- Text preprocessing: cleaning, case folding, tokenization, stopword removal, stemming.
- Feature extraction using:
  - TF-IDF
  - Bag of Words (BoW)
  - Word Embedding
- Sentiment classification using:
  - Logistic Regression
  - Naive Bayes
  - LSTM (Long Short-Term Memory)
- Achieved **99.02% accuracy** using LSTM with Word Embedding.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)&nbsp;
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)&nbsp;
![Keras](https://img.shields.io/badge/-Keras-D00000?style=flat&logo=keras&logoColor=white)&nbsp;
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)&nbsp;
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)&nbsp;
![Seaborn](https://img.shields.io/badge/-Seaborn-5A9FD4?style=flat&logo=seaborn&logoColor=white)&nbsp;
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)&nbsp;

## 🚀 How to Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/viapiyaaa/Project_Sentiment_Analysis_Gojek.git
cd Project_Sentiment_Analysis_Gojek

# 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run app.py
```

## 📊 Sample Result (Optional)  

**Confusion matrix**
- Algoritma Logistic Regression dengan Bag of Words (BoW)
![image](https://github.com/user-attachments/assets/b8457391-b2cc-4548-aed9-0ab0afa755a2)

- Algoritma Logistic Regression dengan TF-IDF
![image](https://github.com/user-attachments/assets/41d67774-f2e2-428e-ac7a-19048657daec)

- Deep Learning LSTM dengan Word Embedding
![image](https://github.com/user-attachments/assets/f0d581b1-1c74-4cb5-b42e-ce87a389fb55)

**Halaman web**
![image](https://github.com/user-attachments/assets/a936b1a2-a307-456c-b20e-82083b5c1139)

## 📄 Dataset
Source: Google Play Store reviews for "Gojek" app
Scraped using: google-play-scraper

## 👩‍💻 Author
Evi Afiyatus Solihah



