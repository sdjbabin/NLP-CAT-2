import streamlit as st
import pandas as pd
import re
import nltk
import spacy
from collections import Counter, defaultdict
from nltk.util import ngrams
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


page_bg = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://res.cloudinary.com/dye74kvqk/image/upload/v1725017362/bg2_yurlw9.png");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://res.cloudinary.com/dye74kvqk/image/upload/v1725017362/bg2_yurlw9.png");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)




# Download necessary NLTK data
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return tokens

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to build bigram model
def build_bigram_model(tokens):
    bigrams = list(ngrams(tokens, 2))
    bigram_freq = Counter(bigrams)
    unigram_freq = Counter(tokens)
    bigram_prob = {bigram: freq / unigram_freq[bigram[0]] for bigram, freq in bigram_freq.items()}
    return bigram_prob

# Function to extract entities with colored output
def extract_and_color_entities(text):
    doc = nlp(text)
    entity_colors = {
        "PRODUCT": "red",
        "ORG": "blue",
        "GPE": "green",
        "PERSON": "orange"
    }
    colored_text = text
    for ent in doc.ents:
        if ent.label_ in entity_colors:
            colored_text = colored_text.replace(ent.text, f'<span style="color:{entity_colors[ent.label_]}">{ent.text}</span>')
    return colored_text

# Function to calculate edit distance
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def extract_proper_nouns(text):
    doc = nlp(text)
    proper_nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return proper_nouns

# Function to suggest correction based on edit distance
def suggest_correction(search_query, product_names, threshold=2):
    suggestions = []
    for product in product_names:
        dist = edit_distance(search_query.lower(), product.lower())
        if dist <= threshold:
            suggestions.append((product, dist))

    if suggestions:
        suggestions.sort(key=lambda x: x[1])
        return [suggestion[0] for suggestion in suggestions]
    else:
        return []

# Function to extract product names
def extract_product_names(text):
    doc = nlp(text)
    product_names = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "PROPN"]]
    return product_names

# Function to extract all product names from DataFrame
def extract_all_product_names(df):
    all_product_names = []
    for review in df['Review']:
        product_names = extract_product_names(review)
        all_product_names.extend(product_names)
    return list(set(all_product_names))

# Function to map product names to reviews
def map_product_names_to_reviews(df, product_names):
    product_to_reviews = defaultdict(list)
    for idx, review in df.iterrows():
        review_text = review['Review']
        for product in product_names:
            if product in review_text:
                product_to_reviews[product].append(review_text)
    return product_to_reviews

# Function to compute cosine similarity
def calculate_cosine_similarity(df):
    vectorizer = TfidfVectorizer()
    review_texts = df['Review'].tolist()
    X = vectorizer.fit_transform(review_texts)
    cosine_sim_matrix = cosine_similarity(X)
    similarity_df = pd.DataFrame(cosine_sim_matrix, index=df.index, columns=df.index)
    return X, similarity_df

# Streamlit application layout
st.title("SportsScribe")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file with reviews. Make sure that the reviews are in the column named Review", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here is a preview of your data:")
    st.write(df.head())

    reviews = df['Review'].fillna('').sum()
    
    # Q1: Standardizing Customer Feedback for Analysis
    st.header("Q1: Standardizing Customer Feedback for Analysis")
    st.write("Processing reviews to clean and tokenize the text...")
    
    tokens = preprocess_text(reviews)
    total_words = len(tokens)
    most_common_word, frequency = Counter(tokens).most_common(1)[0]

    st.write(f"Total number of words: {total_words}")
    st.write(f"Most frequently mentioned word: '{most_common_word}' with a frequency of {frequency}")

    # Q2: Edit Distance for Product Name Correction
    st.header("Q2: Edit Distance for Product Name Correction")
    st.write("Suggest a correct product name based on your input.")

    # Extract product names from reviews
    product_names = extract_all_product_names(df)

    # Input box with real-time suggestions
    search_query = st.text_input("Enter a search query:", key="search_query")
    
    if search_query:
        suggestions = suggest_correction(search_query, product_names)
        if suggestions:
            st.write("Did you mean?:")
            st.write(", ".join(suggestions))
        else:
            st.write("No suggestions available")

    # Q3: N-gram Language Model for Product Recommendations
    st.header("Q3: N-gram Language Model for Product Recommendations")
    st.write("Predict the next word based on bigram probabilities.")

    search_query_bigram = st.text_input("Enter a search query for bigram prediction:")
    tokens = preprocess_text(reviews)
    bigram_prob = build_bigram_model(tokens)

    if search_query_bigram:
        tokens_query = preprocess_text(search_query_bigram)
        last_word = tokens_query[-1]
        candidates = [(w2, prob) for (w1, w2), prob in bigram_prob.items() if w1 == last_word]
        if candidates:
            next_word = max(candidates, key=lambda x: x[1])[0]
            st.write(f"Predicted next word: '{next_word}'")
        else:
            st.write("No prediction available")

    # Q4: Named Entity Recognition (NER) for Reviews
    st.header("Q4: Named Entity Recognition (NER) for Reviews")
    st.write("Extract entities from the reviews with color-coded text.")
    
    for review in df['Review'].head(10):  # Show only a few for display
        colored_text = extract_and_color_entities(review)
        st.markdown(f"{colored_text}", unsafe_allow_html=True)

    # Q5: Product Names Mapping, Cosine Similarity, and Plotting
    st.header("Q5: Product Names Mapping, Cosine Similarity, and Plotting")
    
    # Extract product names
    product_names = extract_all_product_names(df)
    st.write("Extracted Product Names:")
    st.write(product_names)

    # Map product names to reviews
    product_to_reviews = map_product_names_to_reviews(df, product_names)
    st.write("Product to Reviews Mapping:")
    # st.write(product_to_reviews)
    
    # Calculate cosine similarity
    X, similarity_df = calculate_cosine_similarity(df)
    st.write("Cosine Similarity between reviews:")
    st.write(similarity_df)
    
    # Visualization of Product Names
    st.write("Visualizing product embeddings with PCA...")
    
    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
    
    # Annotate points with product names
    for product, reviews in product_to_reviews.items():
        product_indices = [df[df['Review'] == review].index[0] for review in reviews if review in df['Review'].values]
        if product_indices:
            product_coords = np.mean(reduced_embeddings[product_indices], axis=0)
            plt.annotate(product, product_coords, fontsize=12)
    
    plt.title('Product Names in Review Space')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    st.pyplot(plt)
