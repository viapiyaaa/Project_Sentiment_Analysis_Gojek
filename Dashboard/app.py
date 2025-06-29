import streamlit as st
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Import fungsi preprocessing kamu
from preprocessing import cleaningText, casefoldingText, fix_slangwords, filteringText, toSentence

current_dir = os.path.dirname(os.path.abspath(__file__))

# Menggabungkan path untuk tokenizer.pkl
# Naik satu level dari current_dir (dari Dashboard/ ke your_github_repo/)
# Lalu masuk ke folder 'Models'
tokenizer_path = os.path.join(current_dir, '..', 'Models', 'tokenizer.pkl')

# Menggabungkan path untuk model LSTM
model_lstm_path = os.path.join(current_dir, '..', 'Models', 'lstm_sentiment_model.h5')


# Load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Load model LSTM
# (Disarankan pakai @st.cache_resource jika di Streamlit untuk performance)
@st.cache_resource
def load_my_lstm_model(path):
    return load_model(path)

model_lstm = load_my_lstm_model(model_lstm_path)

# Dictionary label
label_dict = {0: "Negatif", 1: "Netral", 2: "Positif"}

# Streamlit UI
# Sidebar
st.set_page_config(page_title="Gojek Sentiment Dashboard", layout="wide")
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="https://lelogama.go-jek.com/cms_editor/2021/05/28/info-gojek-2.png" width="40">
        <h3 style="margin: 0;">Gojek App Dashboard</h3>
    </div>
    """,
    unsafe_allow_html=True
)
page = st.sidebar.radio("Navigate", ["Home", "Analysis", "Sentiment Forecast"]) 

# Main Page
if page == "Home":
    st.markdown("<h1 style='font-size: 40px;'>🛵 Gojek Mobile App Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("""
    Welcome to the **Gojek app user sentiment analysis dashboard**. 
     
    This dashboard aims to understand how **users perceive the Gojek application** based on the reviews they provide.
    
    This review was processed using **automatic sentiment analysis** to group comments into three categories: 💚 **Positive**, ⚪ **Neutral**, and ❤️ **Negative**
    """, unsafe_allow_html=True)

    st.markdown("""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 12px; border-left: 5px solid #388e3c;'>
                    <h4>📌 The main purpose:</h4>
                        <ul>
                            <li>Knowing user perceptions of Gojek application performance</li>
                            <li>Providing insights for Gojek developers and product teams</li>
                            <li>Analyze user comments in an easy-to-understand visual form</li>
                        </ul>
                </div>
""", unsafe_allow_html=True)

elif page == "Analysis":
    st.markdown("## 🗂️ Gojek Review Analysis")
    
    st.markdown("---") 
    
    st.write("The following are the results of reviews from Gojek application users that have been collected and analyzed.")

    df = pd.read_csv("ulasan_gojek (hasil scraping).csv")

    # Konversi score ke sentimen
    def score_to_sentiment(score):
        if score >= 4:
            return "Positif"
        elif score == 3:
            return "Netral"
        else:
            return "Negatif"

    df['sentimen'] = df['score'].apply(score_to_sentiment)
    
    # Hitung total dan distribusi sentimen
    total_reviews = len(df)
    sentiment_counts = df['sentimen'].value_counts()

    # Warna khusus untuk masing-masing sentimen
    warna = {
        'Positif': '#2ecc71',
        'Netral': '#f1c40f',
        'Negatif': '#e74c3c'    
    }

    # Tampilkan metrik dalam bentuk cards
    st.markdown("### 🔢 Sentiment Summary")

    # Tampilkan metrik dengan label bahasa Indonesia
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Positive", f"{sentiment_counts.get('Positif', 0) / total_reviews:.0%}")
    col3.metric("Neutral", f"{sentiment_counts.get('Netral', 0) / total_reviews:.0%}")
    col4.metric("Negative", f"{sentiment_counts.get('Negatif', 0) / total_reviews:.0%}")

    st.subheader("📊 Gojek Comment Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot
    sns.set_style("whitegrid")
    palette = {"Positif": "#27ae60", "Netral": "#f1c40f", "Negatif": "#e74c3c"}

    sns.countplot(data=df, x="sentimen", order=["Positif", "Netral", "Negatif"], palette=palette, ax=ax)

    # Tambahkan label jumlah
    for p in ax.patches:
        value = int(p.get_height())
        ax.annotate(f'{value}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Judul dan gaya
    ax.set_title("Number of Comments per Sentiment", fontsize=16, weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Amount", fontsize=12)
    sns.despine()

    st.pyplot(fig)

    st.markdown("---")
    
    # WordCloud Komentar
    if 'content' in df.columns:
        st.subheader("☁️ WordCloud Comments")
        # Positif
        st.markdown("#### Positive")
        positif_text = ' '.join(df[df['sentimen'] == 'Positif']['content'].astype(str))
        wordcloud_pos = WordCloud(width=800, height=300, background_color='white').generate(positif_text)
        fig1, ax1 = plt.subplots()
        ax1.imshow(wordcloud_pos, interpolation='bilinear')
        ax1.axis("off")
        st.pyplot(fig1)

        # Netral
        st.markdown("#### Neutral")
        netral_text = ' '.join(df[df['sentimen'] == 'Netral']['content'].astype(str))
        wordcloud_netral = WordCloud(width=800, height=300, background_color='white').generate(netral_text)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud_netral, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

        # Negatif
        st.markdown("#### Negative")
        negatif_text = ' '.join(df[df['sentimen'] == 'Negatif']['content'].astype(str))
        wordcloud_neg = WordCloud(width=800, height=300, background_color='white').generate(negatif_text)
        fig3, ax3 = plt.subplots()
        ax3.imshow(wordcloud_neg, interpolation='bilinear')
        ax3.axis("off")
        st.pyplot(fig3)
    else:
        st.info("Kolom 'content' tidak ditemukan di data.")

    st.markdown("### 🗣️ Example of User Review")

    col1, col2, col3 = st.columns(3)

    # Ambil 2 komentar positif
    positif_samples = df[df['sentimen'] == 'Positif']['content'].sample(2, random_state=42)

    with col1:
        for ulasan in positif_samples:
            st.success(f"⭐️⭐️⭐️⭐️⭐️\n\n{ulasan}")
            
    
    netral_samples = df[df['sentimen'] == 'Netral']['content'].sample(2, random_state=12)
    with col2:
        for ulasan in netral_samples:
            st.info(f"⭐️⭐️⭐️\n\n{ulasan}")

    # Ambil 2 komentar negatif
    negatif_samples = df[df['sentimen'] == 'Negatif']['content'].sample(2, random_state=24)

    with col3:
        for ulasan in negatif_samples:
            st.error(f"⭐️⭐️\n\n{ulasan}")

elif page == "Sentiment Forecast":
    st.markdown("""
        <style>
        .big-title {
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
        }
        .analyze-button > button {
            background-color: #00aa13;
            color: white;
            border: none;
            padding: 0.5em 2em;
            font-size: 16px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .analyze-button > button:hover {
            background-color: #008c10;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🧪 Test User Comments")
    st.markdown("---") 
    st.caption("Analyze the sentiment of Gojek user feedback in Bahasa Indonesia")    
    st.markdown("Enter a user comment from the Gojek app to see whether it is classified as 💚 **Positive**, ⚪ **Neutral**, and ❤️ **Negative** based on sentiment analysis.")
    user_input = st.text_area(" ", placeholder="Example: 'Aplikasi ini sangat membantu dan cepat'", height=150)

    col1, col2, _ = st.columns([1,1,2])
    with col1:
        run = st.button("🔍 Analysis", key="analyze-button")
    if run:
        if user_input.strip() == "":
            st.warning("Teks komentar tidak boleh kosong.")
        else:
            # Preprocessing
            kalimat = cleaningText(user_input)
            kalimat = casefoldingText(kalimat)
            kalimat = fix_slangwords(kalimat)
            kalimat = kalimat.split()
            kalimat = filteringText(kalimat)
            kalimat = toSentence(kalimat)

            # Tokenisasi & padding
            sekuens = tokenizer.texts_to_sequences([kalimat])
            padded = pad_sequences(sekuens, maxlen=100)

            # Prediksi
            prediksi = model_lstm.predict(padded)
            label_index = np.argmax(prediksi)
            label = label_dict[label_index]
            confidence = prediksi[0][label_index]

            # Output hasil
            st.markdown("### 📊 Sentiment Analysis Results")
            if label == "Positif":
                st.success(f"✅ **Sentimen:** Positive  \n🎯 **Kepercayaan:** {confidence:.2%}")
            elif label == "Negatif":
                st.error(f"❌ **Sentimen:** Negative  \n🎯 **Kepercayaan:** {confidence:.2%}")
            else:
                st.info(f"⚪ **Sentimen:** Neutral  \n🎯 **Kepercayaan:** {confidence:.2%}")
