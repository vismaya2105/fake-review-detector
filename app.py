import time
import random
import re
import streamlit as st
import joblib
import pickle
import json
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import datetime
import setup

# -------------------------------------------------------
# SCRAPER — built into app
# -------------------------------------------------------
def human_delay(mn=2.0, mx=4.0):
    time.sleep(random.uniform(mn, mx))

def create_driver():
    try:
        import undetected_chromedriver as uc
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        driver = uc.Chrome(options=options)
        return driver
    except Exception:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=options)

def scrape_amazon(url, max_reviews=50):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    reviews = []
    try:
        driver = create_driver()
        if "/dp/" in url and "/product-reviews/" not in url:
            asin = url.split("/dp/")[1].split("/")[0].split("?")[0]
            url = f"https://www.amazon.in/product-reviews/{asin}?reviewerType=all_reviews&pageNumber=1"
        page = 1
        while len(reviews) < max_reviews:
            purl = url if page == 1 else url + f"&pageNumber={page}"
            driver.get(purl)
            human_delay(3, 5)
            try:
                WebDriverWait(driver, 12).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-hook="review-body"]'))
                )
            except Exception:
                break
            els = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review-body"] span')
            if not els:
                break
            c = 0
            for el in els:
                t = el.text.strip()
                if t and len(t) > 15:
                    reviews.append(t)
                    c += 1
                    if len(reviews) >= max_reviews:
                        break
            if c == 0:
                break
            page += 1
            human_delay(2, 4)
        driver.quit()
    except Exception:
        pass
    return reviews

def scrape_flipkart(url, max_reviews=50):
    from selenium.webdriver.common.by import By
    reviews = []
    try:
        driver = create_driver()
        for page in range(1, 6):
            purl = url + f"&page={page}" if "?" in url else url + f"?page={page}"
            driver.get(purl)
            human_delay(3, 5)
            for sel in [".t-ZTKy", "._6K-7Co", ".reviewText"]:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in els:
                    t = el.text.strip()
                    if t and len(t) > 15:
                        reviews.append(t)
                if reviews:
                    break
            if len(reviews) >= max_reviews:
                break
        driver.quit()
    except Exception:
        pass
    return reviews

def get_reviews_from_url(url, max_reviews=50):
    url = url.strip()
    if not url.startswith("http"):
        return [], "Invalid URL."
    if "amazon" in url.lower():
        r = scrape_amazon(url, max_reviews)
        return (r, None) if r else ([], "Could not extract reviews from Amazon. Try Manual Input mode.")
    elif "flipkart" in url.lower():
        r = scrape_flipkart(url, max_reviews)
        return (r, None) if r else ([], "Could not extract reviews from Flipkart. Try Manual Input mode.")
    return [], "Only Amazon and Flipkart URLs are supported."

# -------------------------------------------------------
# SENTIMENT ANALYSIS (VADER)
# -------------------------------------------------------
def get_sentiment(text):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "Positive", round(compound * 100, 1), scores
        elif compound <= -0.05:
            return "Negative", round(abs(compound) * 100, 1), scores
        else:
            return "Neutral", 50.0, scores
    except ImportError:
        words = text.lower().split()
        pos = ["good","great","excellent","amazing","best","love","perfect","awesome","fantastic","wonderful"]
        neg = ["bad","worst","terrible","awful","hate","poor","waste","broken","useless","disappointed"]
        pos_count = sum(1 for w in words if w in pos)
        neg_count = sum(1 for w in words if w in neg)
        if pos_count > neg_count:
            return "Positive", round(min((pos_count / max(len(words), 1)) * 300, 95), 1), {}
        elif neg_count > pos_count:
            return "Negative", round(min((neg_count / max(len(words), 1)) * 300, 95), 1), {}
        else:
            return "Neutral", 50.0, {}

# -------------------------------------------------------
# REVIEWER CREDIBILITY SCORE
# -------------------------------------------------------
def get_credibility_score(text):
    score = 100
    reasons = []
    words = text.lower().split()
    total_words = len(words)

    # Too short
    if total_words < 5:
        score -= 30
        reasons.append("Very short review")
    elif total_words < 15:
        score -= 15
        reasons.append("Short review")

    # Excessive punctuation / caps
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.4:
        score -= 20
        reasons.append("Excessive capitals")

    exclaim = text.count("!")
    if exclaim > 3:
        score -= 15
        reasons.append("Too many exclamation marks")

    # Repetitive words
    unique_ratio = len(set(words)) / max(total_words, 1)
    if unique_ratio < 0.5:
        score -= 20
        reasons.append("Repetitive words")

    # Suspicious phrases
    spam_phrases = [
        "best product ever", "must buy", "highly recommend",
        "five star", "100%", "perfect product", "no complaints",
        "amazing product", "love this product", "great product"
    ]
    text_lower = text.lower()
    for phrase in spam_phrases:
        if phrase in text_lower:
            score -= 10
            reasons.append(f"Contains spam phrase: '{phrase}'")
            break

    # Generic superlatives
    superlatives = ["absolutely", "totally", "completely", "perfectly", "extremely"]
    sup_count = sum(1 for w in words if w in superlatives)
    if sup_count >= 3:
        score -= 15
        reasons.append("Too many superlatives")

    # Good signs
    if total_words > 40:
        score += 10
    if any(c in text for c in [".", ","]):
        score += 5

    score = max(0, min(100, score))

    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"
    else:
        level = "Low"

    return score, level, reasons

# -------------------------------------------------------
# REVIEW HELPFULNESS PREDICTOR
# -------------------------------------------------------
def predict_helpfulness(text):
    score = 0
    words = text.split()
    total_words = len(words)

    # Length score (detailed reviews are more helpful)
    if total_words >= 50:
        score += 30
    elif total_words >= 25:
        score += 20
    elif total_words >= 10:
        score += 10

    # Has specific details
    detail_keywords = [
        "quality", "size", "color", "material", "battery", "camera",
        "screen", "sound", "performance", "speed", "price", "value",
        "delivery", "packaging", "warranty", "feature", "design"
    ]
    detail_count = sum(1 for w in words if w.lower() in detail_keywords)
    score += min(detail_count * 8, 30)

    # Has pros and cons (balanced review)
    pos_words = ["good","great","excellent","love","perfect","amazing","best","works","fast"]
    neg_words = ["but","however","although","except","issue","problem","bad","poor","could be better"]
    has_pos = any(w in text.lower() for w in pos_words)
    has_neg = any(w in text.lower() for w in neg_words)
    if has_pos and has_neg:
        score += 20

    # Proper sentences
    sentences = text.count(".") + text.count("!") + text.count("?")
    if sentences >= 2:
        score += 10

    # Penalize very short or spammy
    if total_words < 5:
        score -= 20
    exclaim = text.count("!")
    if exclaim > 4:
        score -= 10

    score = max(0, min(100, score))

    if score >= 70:
        label = "Very Helpful"
        color = "#22c55e"
    elif score >= 40:
        label = "Somewhat Helpful"
        color = "#f59e0b"
    else:
        label = "Not Helpful"
        color = "#ef4444"

    return score, label, color

# -------------------------------------------------------
# PRODUCT QUALITY SCORE
# -------------------------------------------------------
def get_product_quality_score(results):
    genuine_reviews = [r for r in results if r["label"] == "Genuine"]
    if not genuine_reviews:
        return 0, "No genuine reviews found", []

    sentiments = []
    for r in genuine_reviews:
        sentiment, score, _ = get_sentiment(r["review"])
        sentiments.append((sentiment, score))

    pos = sum(1 for s, _ in sentiments if s == "Positive")
    neg = sum(1 for s, _ in sentiments if s == "Negative")
    neu = sum(1 for s, _ in sentiments if s == "Neutral")
    total = len(sentiments)

    quality_score = round(((pos * 100) + (neu * 60) + (neg * 20)) / max(total, 1), 1)
    quality_score = min(100, quality_score)

    if quality_score >= 75:
        verdict = "Excellent Product"
    elif quality_score >= 55:
        verdict = "Good Product"
    elif quality_score >= 35:
        verdict = "Average Product"
    else:
        verdict = "Poor Product"

    breakdown = [
        {"Sentiment": "Positive", "Count": pos,  "Percentage": round(pos/total*100, 1)},
        {"Sentiment": "Neutral",  "Count": neu,  "Percentage": round(neu/total*100, 1)},
        {"Sentiment": "Negative", "Count": neg,  "Percentage": round(neg/total*100, 1)},
    ]
    return quality_score, verdict, breakdown

# -------------------------------------------------------
# PRICE VS QUALITY SCORE
# -------------------------------------------------------
def get_price_quality_score(quality_score, trust_score, price_range):
    # Base score from quality and trust
    base = (quality_score * 0.6) + (trust_score * 0.4)

    # Adjust based on price range
    price_multiplier = {
        "Budget (Under 500)":      1.15,
        "Mid-range (500 - 2000)":  1.0,
        "Premium (2000 - 10000)":  0.9,
        "Luxury (Above 10000)":    0.8,
    }
    multiplier = price_multiplier.get(price_range, 1.0)
    final_score = round(min(base * multiplier, 100), 1)

    if final_score >= 75:
        verdict = "Excellent Value for Money"
        color   = "#22c55e"
    elif final_score >= 55:
        verdict = "Good Value for Money"
        color   = "#3b82f6"
    elif final_score >= 35:
        verdict = "Average Value for Money"
        color   = "#f59e0b"
    else:
        verdict = "Poor Value for Money"
        color   = "#ef4444"

    return final_score, verdict, color


# -------------------------------------------------------
# BUY / DON'T BUY RECOMMENDATION
# -------------------------------------------------------
def get_buy_recommendation(fake_pct, trust_score, quality_score, price_quality_score):
    reasons_buy     = []
    reasons_avoid   = []

    if trust_score >= 70:
        reasons_buy.append(f"High trust score ({trust_score}%) — reviews appear genuine")
    elif trust_score < 40:
        reasons_avoid.append(f"Low trust score ({trust_score}%) — many fake reviews detected")

    if quality_score >= 70:
        reasons_buy.append(f"Good product quality ({quality_score}%) based on genuine reviews")
    elif quality_score < 40:
        reasons_avoid.append(f"Poor product quality ({quality_score}%) based on genuine reviews")

    if fake_pct < 20:
        reasons_buy.append(f"Only {fake_pct}% fake reviews — mostly authentic feedback")
    elif fake_pct > 50:
        reasons_avoid.append(f"{fake_pct}% of reviews are fake — unreliable product feedback")

    if price_quality_score >= 70:
        reasons_buy.append("Good value for the price range")
    elif price_quality_score < 35:
        reasons_avoid.append("Poor value for the price range")

    # Final decision
    buy_score = len(reasons_buy)
    avoid_score = len(reasons_avoid)

    if buy_score > avoid_score and trust_score >= 50:
        decision = "BUY"
        decision_color = "#22c55e"
        decision_emoji = "✅"
    elif avoid_score > buy_score or trust_score < 30:
        decision = "AVOID"
        decision_color = "#ef4444"
        decision_emoji = "❌"
    else:
        decision = "CONSIDER WITH CAUTION"
        decision_color = "#f59e0b"
        decision_emoji = "⚠️"

    return decision, decision_color, decision_emoji, reasons_buy, reasons_avoid


# -------------------------------------------------------
# REVIEW SUMMARY GENERATOR
# -------------------------------------------------------
def generate_review_summary(results, product_name="this product"):
    genuine_reviews = [r for r in results if r["label"] == "Genuine"]
    fake_count      = sum(1 for r in results if r["label"] == "Fake")
    total           = len(results)

    if not genuine_reviews:
        return f"No genuine reviews found for {product_name}. All {total} reviews appear to be fake."

    # Sentiment counts
    pos = sum(1 for r in genuine_reviews if r["sentiment"] == "Positive")
    neg = sum(1 for r in genuine_reviews if r["sentiment"] == "Negative")
    neu = sum(1 for r in genuine_reviews if r["sentiment"] == "Neutral")

    # Common words from genuine reviews
    all_words = " ".join([r["review"] for r in genuine_reviews]).lower().split()
    stopwords = {"the","a","an","is","it","this","that","and","or","but","in","on","at",
                 "to","for","of","with","i","my","was","are","be","have","has","not","very",
                 "so","its","also","just","get","got","good","bad","product","item"}
    word_freq = {}
    for w in all_words:
        w = re.sub(r"[^a-z]", "", w)
        if w and w not in stopwords and len(w) > 3:
            word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

    # Build summary
    fake_pct = round((fake_count / total) * 100, 1)
    summary_parts = []

    # Line 1 - fake review situation
    if fake_pct > 50:
        summary_parts.append(f"Warning: {fake_pct}% of reviews for {product_name} appear to be fake, making it difficult to assess the true quality.")
    elif fake_pct > 20:
        summary_parts.append(f"{product_name} has {fake_pct}% suspicious reviews, but {len(genuine_reviews)} genuine reviews were found.")
    else:
        summary_parts.append(f"{product_name} has mostly genuine reviews with only {fake_pct}% flagged as suspicious.")

    # Line 2 - sentiment
    if pos > neg:
        summary_parts.append(f"Genuine buyers are largely positive — {pos} out of {len(genuine_reviews)} real reviews express satisfaction.")
    elif neg > pos:
        summary_parts.append(f"Genuine buyers express concerns — {neg} out of {len(genuine_reviews)} real reviews are negative.")
    else:
        summary_parts.append(f"Genuine buyer opinions are mixed with {pos} positive and {neg} negative reviews.")

    # Line 3 - top keywords
    if top_words:
        summary_parts.append(f"Reviewers most commonly mention: {', '.join(top_words[:4])}.")

    return " ".join(summary_parts)



# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Fake Review Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-value  { font-size: 32px; font-weight: bold; margin: 8px 0; }
.metric-label  { font-size: 13px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
.fake-color    { color: #ef4444; }
.genuine-color { color: #22c55e; }
.trust-color   { color: #3b82f6; }
.warn-color    { color: #f59e0b; }
.review-card {
    background: #1f2937;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.genuine-card {
    background: #1f2937;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.section-header {
    font-size: 20px; font-weight: 600;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODELS AND VECTORIZER
# -------------------------------------------------------
@st.cache_resource
def load_all_models():
    models = {}
    model_files = {
        "Random Forest":       "model_random_forest.pkl",
        "Logistic Regression": "model_logistic_regression.pkl",
        "SVM":                 "model_svm_linearsvc.pkl",
        "XGBoost":             "model_xgboost.pkl",
    }
    for name, fname in model_files.items():
        if os.path.exists(fname):
            try:
                models[name] = pickle.load(open(fname, "rb"))
            except Exception:
                pass
    return models

@st.cache_resource
def load_vectorizer():
    return pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@st.cache_data
def load_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json") as f:
            return json.load(f)
    return {}

models     = load_all_models()
vectorizer = load_vectorizer()
metrics    = load_metrics()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("🔎 Fake Review Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔍 Analyzer",
    "📊 Dashboard",
    "🧠 Deep Analysis",
    "🤖 Model Comparison",
    "📄 Report"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
available_models = list(models.keys())
if not available_models:
    st.error("No model .pkl files found!")
    st.stop()
selected_model_name = st.sidebar.selectbox("Choose ML Model", available_models)
selected_model = models[selected_model_name]
max_reviews = st.sidebar.slider("Max reviews to scrape", 10, 100, 50, 10)
st.sidebar.markdown("---")
if metrics.get("_best_model"):
    st.sidebar.success(f"🏆 Best model: {metrics['_best_model']}")

# -------------------------------------------------------
# HELPER: ANALYZE REVIEWS
# -------------------------------------------------------
def analyze_reviews(reviews, model, vec):
    fake, genuine = 0, 0
    results = []
    for review in reviews:
        if not review.strip():
            continue
        text_vec = vec.transform([review])
        final    = hstack([text_vec, [[5]]])
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(final)[0]
            pred       = int(np.argmax(proba))
            confidence = round(float(np.max(proba)) * 100, 1)
        else:
            pred = int(model.predict(final)[0])
            try:
                score      = model.decision_function(final)[0]
                confidence = round(min(abs(float(score)) * 20 + 50, 99), 1)
            except Exception:
                confidence = 0.0
        label = "Genuine" if pred == 1 else "Fake"
        sentiment, sent_score, _ = get_sentiment(review)
        cred_score, cred_level, cred_reasons = get_credibility_score(review)
        help_score, help_label, help_color    = predict_helpfulness(review)
        if label == "Fake":
            fake += 1
        else:
            genuine += 1
        results.append({
            "review":       review,
            "label":        label,
            "confidence":   confidence,
            "sentiment":    sentiment,
            "sent_score":   sent_score,
            "cred_score":   cred_score,
            "cred_level":   cred_level,
            "cred_reasons": cred_reasons,
            "help_score":   help_score,
            "help_label":   help_label,
            "help_color":   help_color,
        })
    total       = fake + genuine
    fake_pct    = round((fake / total) * 100, 2) if total > 0 else 0
    trust_score = round(100 - fake_pct, 2)
    return results, fake, genuine, total, fake_pct, trust_score

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if page == "🏠 Home":
    st.markdown("# 🔎 AI Fake Review Detection Platform")
    st.markdown("#### Detect fake reviews + sentiment + credibility + helpfulness + product quality")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>ML Models</div>
            <div class='metric-value trust-color'>{len(models)}</div>
            <div class='metric-label'>trained models</div></div>""", unsafe_allow_html=True)
    with col2:
        best = metrics.get("_best_model", "N/A")
        bacc = metrics.get(best, {}).get("accuracy", "—") if best != "N/A" else "—"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Best Accuracy</div>
            <div class='metric-value genuine-color'>{bacc}%</div>
            <div class='metric-label'>{best}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Analysis Features</div>
            <div class='metric-value warn-color'>5</div>
            <div class='metric-label'>prediction types</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Platforms</div>
            <div class='metric-value fake-color'>2</div>
            <div class='metric-label'>Amazon + Flipkart</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧠 What this system predicts")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.success("**Fake Detection**\nReal vs Fake review using 4 ML models")
    with c2:
        st.info("**Sentiment**\nPositive / Negative / Neutral score")
    with c3:
        st.warning("**Credibility**\nHow trustworthy is the reviewer")
    with c4:
        st.info("**Helpfulness**\nWill this review help buyers")
    with c5:
        st.success("**Product Quality**\nOverall score from genuine reviews")
    st.markdown("---")
    st.markdown("### 🚀 How to use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1** — Go to 🔍 Analyzer → paste reviews manually or use URL")
    with c2:
        st.info("**Step 2** — Select ML model from sidebar and click Analyze")
    with c3:
        st.info("**Step 3** — View 📊 Dashboard and 🧠 Deep Analysis for full results")

# -------------------------------------------------------
# ANALYZER PAGE
# -------------------------------------------------------
elif page == "🔍 Analyzer":
    st.markdown("# 🔍 Review Analyzer")
    st.markdown(f"Using model: **{selected_model_name}**")
    st.markdown("---")
    # Product details inputs
    st.markdown("### 📦 Product Details (optional but recommended)")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        product_name = st.text_input("Product Name", placeholder="e.g. boAt Airdopes 141")
    with col_b:
        product_category = st.text_input("Category", placeholder="e.g. Earphones, Mobile, Laptop")
    with col_c:
        price_range = st.selectbox("Price Range", [
            "Budget (Under 500)",
            "Mid-range (500 - 2000)",
            "Premium (2000 - 10000)",
            "Luxury (Above 10000)"
        ])
    st.session_state["product_name"]     = product_name if product_name else "This Product"
    st.session_state["product_category"] = product_category if product_category else "General"
    st.session_state["price_range"]      = price_range
    st.markdown("---")

    mode = st.radio("Choose input method",
        ["🌐 Product URL (Amazon / Flipkart)", "✏️ Manual Review Input", "📁 Upload CSV File"],
        horizontal=True)
    reviews = []

    if mode == "🌐 Product URL (Amazon / Flipkart)":
        st.markdown("##### Paste the product page URL below")
        url = st.text_input("Product URL", placeholder="https://www.amazon.in/dp/...")
        extract_btn = st.button("🔍 Extract & Analyze", type="primary")
        if extract_btn and url:
            with st.spinner("Opening browser and extracting reviews... (30–60 seconds)"):
                try:
                    reviews, error = get_reviews_from_url(url, max_reviews=max_reviews)
                    if error:
                        st.error(f"Scraper error: {error}")
                        reviews = []
                    elif reviews:
                        st.success(f"✅ Extracted {len(reviews)} reviews!")
                    else:
                        st.warning("No reviews found. Try Manual Input mode.")
                except Exception as e:
                    st.error(f"Scraper failed: {str(e)}")

    elif mode == "✏️ Manual Review Input":
        st.markdown("##### Paste reviews below — one review per line")
        text = st.text_area("Reviews", height=200,
            placeholder="Great product, works perfectly!\nTerrible quality, broke after 2 days.")
        if st.button("🔍 Analyze Reviews", type="primary"):
            reviews = [r.strip() for r in text.split("\n") if r.strip()]
            if not reviews:
                st.warning("Please paste at least one review.")

    else:
        st.markdown("##### Upload a CSV file with a column named 'review_text'")
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                if "review_text" in df_upload.columns:
                    reviews = df_upload["review_text"].dropna().tolist()
                    st.success(f"✅ Loaded {len(reviews)} reviews from CSV!")
                else:
                    col = st.selectbox("Select the review column", df_upload.columns.tolist())
                    reviews = df_upload[col].dropna().tolist()
                    st.success(f"✅ Loaded {len(reviews)} reviews!")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

    if reviews:
        with st.spinner(f"Running full analysis on {len(reviews)} reviews..."):
            results, fake, genuine, total, fake_pct, trust_score = analyze_reviews(
                reviews, selected_model, vectorizer)
        st.session_state["results"]     = results
        st.session_state["fake"]        = fake
        st.session_state["genuine"]     = genuine
        st.session_state["total"]       = total
        st.session_state["fake_pct"]    = fake_pct
        st.session_state["trust_score"] = trust_score
        st.session_state["model_used"]  = selected_model_name
        st.session_state["reviews"]     = reviews
        quality_score, verdict, breakdown = get_product_quality_score(results)
        st.session_state["quality_score"] = quality_score
        st.session_state["verdict"]       = verdict
        st.session_state["breakdown"]     = breakdown

        # Price vs quality
        pq_score, pq_verdict, pq_color = get_price_quality_score(
            quality_score, trust_score, price_range)
        st.session_state["pq_score"]   = pq_score
        st.session_state["pq_verdict"] = pq_verdict
        st.session_state["pq_color"]   = pq_color

        # Buy recommendation
        decision, dec_color, dec_emoji, reasons_buy, reasons_avoid = get_buy_recommendation(
            fake_pct, trust_score, quality_score, pq_score)
        st.session_state["decision"]      = decision
        st.session_state["dec_color"]     = dec_color
        st.session_state["dec_emoji"]     = dec_emoji
        st.session_state["reasons_buy"]   = reasons_buy
        st.session_state["reasons_avoid"] = reasons_avoid

        # Review summary
        pname = st.session_state.get("product_name", "This Product")
        summary = generate_review_summary(results, pname)
        st.session_state["review_summary"] = summary

        st.success("✅ Full analysis complete! Go to 📊 Dashboard and 🧠 Deep Analysis.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews", total)
        c2.metric("Fake Reviews",    fake,    delta=f"{fake_pct}%",       delta_color="inverse")
        c3.metric("Genuine Reviews", genuine, delta=f"{100-fake_pct}%")
        c4.metric("Trust Score",     f"{trust_score}%")

# -------------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------------
elif page == "📊 Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    if "results" not in st.session_state:
        st.warning("⚠️ No analysis found. Please go to 🔍 Analyzer first.")
        st.stop()
    fake          = st.session_state["fake"]
    genuine       = st.session_state["genuine"]
    total         = st.session_state["total"]
    fake_pct      = st.session_state["fake_pct"]
    trust_score   = st.session_state["trust_score"]
    results       = st.session_state["results"]
    model_used    = st.session_state.get("model_used", "Unknown")
    quality_score = st.session_state.get("quality_score", 0)
    verdict       = st.session_state.get("verdict", "N/A")
    pname         = st.session_state.get("product_name", "")
    pcategory     = st.session_state.get("product_category", "")
    pq_score      = st.session_state.get("pq_score", 0)
    pq_verdict    = st.session_state.get("pq_verdict", "N/A")
    pq_color      = st.session_state.get("pq_color", "#9ca3af")
    decision      = st.session_state.get("decision", "N/A")
    dec_color     = st.session_state.get("dec_color", "#9ca3af")
    dec_emoji     = st.session_state.get("dec_emoji", "")
    reasons_buy   = st.session_state.get("reasons_buy", [])
    reasons_avoid = st.session_state.get("reasons_avoid", [])
    summary       = st.session_state.get("review_summary", "")

    # Product header
    if pname:
        st.markdown(f"## 📦 {pname}")
        if pcategory:
            st.markdown(f"**Category:** {pcategory}")
    st.markdown(f"*Analysis using: **{model_used}***")

    # AI Summary box
    if summary:
        st.markdown("---")
        st.markdown("### 📝 AI Review Summary")
        st.info(summary)

    # Reasons to buy and avoid
    if reasons_buy or reasons_avoid:
        st.markdown("---")
        st.markdown("### 🛒 Product Analysis")
        col_buy, col_avoid = st.columns(2)
        with col_buy:
            st.markdown("**✅ Reasons to Buy:**")
            if reasons_buy:
                for r in reasons_buy:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- No strong reasons found")
        with col_avoid:
            st.markdown("**❌ Reasons to Avoid:**")
            if reasons_avoid:
                for r in reasons_avoid:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- No major concerns found")

    st.markdown("---")

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Total Reviews</div>
            <div class='metric-value warn-color'>{total}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Fake Reviews</div>
            <div class='metric-value fake-color'>{fake} ({fake_pct}%)</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Genuine Reviews</div>
            <div class='metric-value genuine-color'>{genuine}</div></div>""", unsafe_allow_html=True)
    with c4:
        color = "genuine-color" if trust_score >= 70 else ("warn-color" if trust_score >= 40 else "fake-color")
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Trust Score</div>
            <div class='metric-value {color}'>{trust_score}%</div></div>""", unsafe_allow_html=True)
    with c5:
        qcolor = "genuine-color" if quality_score >= 70 else ("warn-color" if quality_score >= 40 else "fake-color")
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Product Quality</div>
            <div class='metric-value {qcolor}'>{quality_score}%</div>
            <div class='metric-label'>{verdict}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if fake_pct < 20:
        st.success("🟢 LOW RISK — This product has mostly genuine reviews")
    elif fake_pct < 40:
        st.warning("🟡 MEDIUM RISK — Some fake reviews detected, proceed with caution")
    else:
        st.error("🔴 HIGH RISK — Large number of fake reviews detected!")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Fake vs Genuine</div>", unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Fake","Genuine"], values=[fake, genuine], hole=0.55,
            marker_colors=["#ef4444","#22c55e"], textinfo="label+percent", textfont_size=14)])
        fig_donut.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
        st.plotly_chart(fig_donut, use_container_width=True)
    with col2:
        st.markdown("<div class='section-header'>Sentiment breakdown</div>", unsafe_allow_html=True)
        sent_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for r in results:
            sent_counts[r["sentiment"]] = sent_counts.get(r["sentiment"], 0) + 1
        fig_sent = go.Figure(data=[go.Pie(
            labels=list(sent_counts.keys()), values=list(sent_counts.values()), hole=0.55,
            marker_colors=["#22c55e","#ef4444","#f59e0b"], textinfo="label+percent", textfont_size=14)])
        fig_sent.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
        st.plotly_chart(fig_sent, use_container_width=True)

    # Trust score gauge
    st.markdown("<div class='section-header'>Product trust score</div>", unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=trust_score, delta={"reference": 70},
        gauge={"axis": {"range": [0,100]}, "bar": {"color": "#3b82f6"},
               "steps": [{"range":[0,40],"color":"#450a0a"},
                         {"range":[40,70],"color":"#451a03"},
                         {"range":[70,100],"color":"#052e16"}]},
        number={"suffix":"%","font":{"color":"white","size":40}}))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=280, margin=dict(t=20,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Word cloud
    st.markdown("<div class='section-header'>Review word cloud</div>", unsafe_allow_html=True)
    all_text = " ".join([r["review"] for r in results])
    if all_text.strip():
        wc = WordCloud(width=1200, height=400, background_color="black", colormap="RdYlGn", max_words=100).generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(14,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_facecolor("black")
        st.pyplot(fig_wc)

    # Suspicious reviews
    st.markdown("<div class='section-header'>Suspicious reviews</div>", unsafe_allow_html=True)
    fake_reviews = [r for r in results if r["label"] == "Fake"]
    if fake_reviews:
        for r in fake_reviews:
            conf_color = "#ef4444" if r["confidence"] > 75 else "#f59e0b"
            st.markdown(f"""<div class='review-card'>
                <span style='color:#9ca3af;font-size:12px;'>FAKE</span>
                <span style='background:{conf_color}22;color:{conf_color};padding:2px 10px;border-radius:20px;font-size:12px;margin-left:6px;'>{r['confidence']}% confidence</span>
                <span style='background:#3b82f622;color:#3b82f6;padding:2px 10px;border-radius:20px;font-size:12px;margin-left:6px;'>{r['sentiment']}</span>
                <span style='background:#ffffff11;color:#9ca3af;padding:2px 10px;border-radius:20px;font-size:12px;margin-left:6px;'>Credibility: {r['cred_level']}</span>
                <p style='margin:8px 0 0 0;color:#e5e7eb;'>{r['review'][:300]}</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("No fake reviews detected!")

    with st.expander("✅ View genuine reviews"):
        for r in [r for r in results if r["label"] == "Genuine"]:
            st.markdown(f"""<div class='genuine-card'>
                <span style='color:#22c55e;font-size:12px;'>GENUINE</span>
                <span style='background:#22c55e22;color:#22c55e;padding:2px 10px;border-radius:20px;font-size:12px;margin-left:6px;'>{r['confidence']}% confidence</span>
                <span style='background:#3b82f622;color:#3b82f6;padding:2px 10px;border-radius:20px;font-size:12px;margin-left:6px;'>{r['sentiment']}</span>
                <p style='margin:8px 0 0 0;color:#e5e7eb;'>{r['review'][:300]}</p>
            </div>""", unsafe_allow_html=True)

# -------------------------------------------------------
# DEEP ANALYSIS PAGE
# -------------------------------------------------------
elif page == "🧠 Deep Analysis":
    st.markdown("# 🧠 Deep Analysis")
    st.markdown("Sentiment · Credibility · Helpfulness · Product Quality")
    st.markdown("---")
    if "results" not in st.session_state:
        st.warning("⚠️ No analysis found. Please go to 🔍 Analyzer first.")
        st.stop()
    results       = st.session_state["results"]
    quality_score = st.session_state.get("quality_score", 0)
    verdict       = st.session_state.get("verdict", "N/A")
    breakdown     = st.session_state.get("breakdown", [])

    # --- PRODUCT QUALITY ---
    st.markdown("## 🏆 Product Quality Score")
    col1, col2 = st.columns([1, 2])
    with col1:
        qcolor = "#22c55e" if quality_score >= 70 else ("#f59e0b" if quality_score >= 40 else "#ef4444")
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Quality Score</div>
            <div class='metric-value' style='color:{qcolor};'>{quality_score}%</div>
            <div class='metric-label'>{verdict}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Based on genuine reviews only")
    with col2:
        if breakdown:
            df_b = pd.DataFrame(breakdown)
            fig_b = px.bar(df_b, x="Sentiment", y="Count", color="Sentiment",
                color_discrete_map={"Positive":"#22c55e","Neutral":"#f59e0b","Negative":"#ef4444"},
                text="Percentage")
            fig_b.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", showlegend=False, height=250, margin=dict(t=10,b=10))
            st.plotly_chart(fig_b, use_container_width=True)
    st.markdown("---")

    # --- SENTIMENT ANALYSIS ---
    st.markdown("## 💬 Sentiment Analysis")
    df_results = pd.DataFrame(results)
    sent_fig = px.histogram(df_results, x="sentiment", color="sentiment",
        color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"},
        title="Sentiment Distribution")
    sent_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", showlegend=False, height=300)
    st.plotly_chart(sent_fig, use_container_width=True)
    st.markdown("---")

    # --- CREDIBILITY ANALYSIS ---
    st.markdown("## 🔐 Reviewer Credibility Analysis")
    cred_counts = {"High": 0, "Medium": 0, "Low": 0}
    for r in results:
        cred_counts[r["cred_level"]] = cred_counts.get(r["cred_level"], 0) + 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>High Credibility</div>
            <div class='metric-value genuine-color'>{cred_counts['High']}</div>
            <div class='metric-label'>reviews</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Medium Credibility</div>
            <div class='metric-value warn-color'>{cred_counts['Medium']}</div>
            <div class='metric-label'>reviews</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Low Credibility</div>
            <div class='metric-value fake-color'>{cred_counts['Low']}</div>
            <div class='metric-label'>reviews</div></div>""", unsafe_allow_html=True)
    st.markdown("---")

    # --- HELPFULNESS ANALYSIS ---
    st.markdown("## 🤝 Review Helpfulness Predictor")
    help_counts = {"Very Helpful": 0, "Somewhat Helpful": 0, "Not Helpful": 0}
    for r in results:
        help_counts[r["help_label"]] = help_counts.get(r["help_label"], 0) + 1
    fig_help = go.Figure(data=[go.Bar(
        x=list(help_counts.keys()), y=list(help_counts.values()),
        marker_color=["#22c55e","#f59e0b","#ef4444"],
        text=list(help_counts.values()), textposition="outside"
    )])
    fig_help.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig_help, use_container_width=True)
    st.markdown("---")

    # --- PER REVIEW TABLE ---
    st.markdown("## 📋 Per Review Full Analysis")
    table_data = []
    for r in results:
        table_data.append({
            "Review":      r["review"][:80] + "..." if len(r["review"]) > 80 else r["review"],
            "Fake/Genuine":r["label"],
            "Confidence":  f"{r['confidence']}%",
            "Sentiment":   r["sentiment"],
            "Credibility": r["cred_level"],
            "Helpfulness": r["help_label"],
        })
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

# -------------------------------------------------------
# MODEL COMPARISON PAGE
# -------------------------------------------------------
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 ML Model Comparison")
    st.markdown("---")
    if not metrics or len(metrics) <= 1:
        st.warning("model_metrics.json not found.")
        st.stop()
    rows = []
    for name, m in metrics.items():
        if name.startswith("_"):
            continue
        rows.append({"Model": name, "Accuracy": m.get("accuracy",0), "F1 Score": m.get("f1_score",0),
                     "Precision": m.get("precision",0), "Recall": m.get("recall",0)})
    df_metrics = pd.DataFrame(rows)
    best = metrics.get("_best_model","")
    st.info(f"🏆 Best performing model: **{best}**")
    st.dataframe(df_metrics.style.highlight_max(
        subset=["Accuracy","F1 Score","Precision","Recall"], color="#052e16"
    ).format("{:.2f}", subset=["Accuracy","F1 Score","Precision","Recall"]),
        use_container_width=True, hide_index=True)
    fig_bar = px.bar(df_metrics, x="Model", y="Accuracy", color="Model", text="Accuracy",
        color_discrete_sequence=["#3b82f6","#22c55e","#f59e0b","#ef4444"])
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", showlegend=False, height=350, yaxis=dict(range=[0,110]))
    st.plotly_chart(fig_bar, use_container_width=True)
    categories = ["Accuracy","F1 Score","Precision","Recall"]
    fig_radar = go.Figure()
    colors = ["#3b82f6","#22c55e","#f59e0b","#ef4444"]
    for i, row in df_metrics.iterrows():
        vals = [row[c] for c in categories] + [row[categories[0]]]
        fig_radar.add_trace(go.Scatterpolar(r=vals, theta=categories+[categories[0]],
            fill="toself", name=row["Model"], line_color=colors[i % len(colors)], opacity=0.7))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
    st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------------------------------
# REPORT PAGE
# -------------------------------------------------------
elif page == "📄 Report":
    st.markdown("# 📄 Download Analysis Report")
    st.markdown("---")
    if "results" not in st.session_state:
        st.warning("⚠️ No analysis found. Please go to 🔍 Analyzer first.")
        st.stop()
    fake          = st.session_state["fake"]
    genuine       = st.session_state["genuine"]
    total         = st.session_state["total"]
    fake_pct      = st.session_state["fake_pct"]
    trust_score   = st.session_state["trust_score"]
    results       = st.session_state["results"]
    model_used    = st.session_state.get("model_used","Unknown")
    quality_score = st.session_state.get("quality_score", 0)
    verdict       = st.session_state.get("verdict","N/A")
    col1, col2 = st.columns(2)
    col1.metric("Total Reviews",  total)
    col1.metric("Fake Reviews",   fake)
    col2.metric("Genuine Reviews",genuine)
    col2.metric("Trust Score",    f"{trust_score}%")
    st.markdown(f"**Model used:** {model_used}")
    st.markdown(f"**Product Quality:** {quality_score}% - {verdict}")
    st.markdown(f"**Date:** {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}")
    if st.button("📥 Generate & Download PDF Report", type="primary"):
        with st.spinner("Generating PDF..."):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Helvetica","B",20)
                pdf.cell(0,12,"AI Fake Review Detection Report",ln=True,align="C")
                pdf.ln(4)
                pdf.set_font("Helvetica","",11)
                pdf.set_text_color(100,100,100)
                pdf.cell(0,8,f"Generated: {datetime.datetime.now().strftime('%d %B %Y %H:%M')}",ln=True,align="C")
                pdf.cell(0,8,f"Model used: {model_used}",ln=True,align="C")
                pdf.ln(6)
                pdf.set_fill_color(240,240,240)
                pdf.set_font("Helvetica","B",13)
                pdf.set_text_color(30,30,30)
                pdf.cell(0,10,"Summary",ln=True,fill=True)
                pdf.ln(2)
                summary = [
                    ("Total Reviews",    str(total)),
                    ("Fake Reviews",     f"{fake} ({fake_pct}%)"),
                    ("Genuine Reviews",  f"{genuine}"),
                    ("Trust Score",      f"{trust_score}%"),
                    ("Product Quality",  f"{quality_score}% - {verdict}"),
                    ("Risk Level",       "LOW" if fake_pct<20 else ("MEDIUM" if fake_pct<40 else "HIGH")),
                ]
                for label, value in summary:
                    pdf.set_font("Helvetica","",12)
                    pdf.set_text_color(80,80,80)
                    pdf.cell(90,9,label+":",ln=False)
                    pdf.set_text_color(30,30,30)
                    pdf.set_font("Helvetica","B",12)
                    pdf.cell(0,9,str(value).encode("latin-1","replace").decode("latin-1"),ln=True)
                pdf.ln(6)
                if metrics and len(metrics) > 1:
                    pdf.set_fill_color(240,240,240)
                    pdf.set_font("Helvetica","B",13)
                    pdf.set_text_color(30,30,30)
                    pdf.cell(0,10,"Model Performance Metrics",ln=True,fill=True)
                    pdf.ln(2)
                    pdf.set_font("Helvetica","B",10)
                    for col_n,w in [("Model",55),("Accuracy",32),("F1",28),("Precision",36),("Recall",30)]:
                        pdf.cell(w,8,col_n,border=1,align="C")
                    pdf.ln()
                    pdf.set_font("Helvetica","",10)
                    for mname,mdata in metrics.items():
                        if mname.startswith("_"):
                            continue
                        pdf.cell(55,8,mname,border=1)
                        pdf.cell(32,8,f"{mdata.get('accuracy',0)}%",border=1,align="C")
                        pdf.cell(28,8,f"{mdata.get('f1_score',0)}%",border=1,align="C")
                        pdf.cell(36,8,f"{mdata.get('precision',0)}%",border=1,align="C")
                        pdf.cell(30,8,f"{mdata.get('recall',0)}%",border=1,align="C")
                        pdf.ln()
                    pdf.ln(6)
                pdf.set_fill_color(240,240,240)
                pdf.set_font("Helvetica","B",13)
                pdf.set_text_color(30,30,30)
                pdf.cell(0,10,f"Detected Fake Reviews ({fake} total)",ln=True,fill=True)
                pdf.ln(2)
                pdf.set_font("Helvetica","",10)
                pdf.set_text_color(60,60,60)
                for i,r in enumerate([r for r in results if r["label"]=="Fake"][:30],1):
                    line = f"{i}. [Conf:{r['confidence']}% | {r['sentiment']} | Cred:{r['cred_level']}] {r['review'][:180]}"
                    line = line.encode('latin-1','replace').decode('latin-1')
                    pdf.multi_cell(0,7,line)
                    pdf.ln(1)
                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    tmp_path = tmp.name
                with open(tmp_path,"rb") as f:
                    pdf_bytes = f.read()
                fname = f"fake_review_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(label="📥 Click here to download your PDF",
                    data=pdf_bytes,file_name=fname,mime="application/pdf")
                st.success("✅ PDF ready!")
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
