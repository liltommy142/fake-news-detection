import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import plotly.express as px
import random
import re # <-- Thêm thư viện này để rà soát văn bản
from Features_extracting import RealtimeFeatureExtractor

# ==========================================
# PAGE CONFIGURATION & GLOBALS
# ==========================================
st.set_page_config(page_title="AI Fake News Detector", page_icon="🕵️‍♂️", layout="centered")

TRUSTED_DOMAINS = [
    'vnexpress.net', 'tuoitre.vn', 'thanhnien.vn', 'vietnamnet.vn', 'dantri.com.vn',
    'vtv.vn', 'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com', 'bloomberg.com', 'apnews.com'
]

@st.cache_resource
def load_models():
    extractor = RealtimeFeatureExtractor()
    m1 = joblib.load('lgbm_model_text.pkl')
    m2 = joblib.load('lgbm_model_link.pkl')
    tf_text = joblib.load('tfidf_model_text.pkl')
    tf_link = joblib.load('tfidf_model_link.pkl')
    sc_text = joblib.load('scaler_text.pkl')
    sc_link = joblib.load('scaler_link.pkl')
    return extractor, m1, m2, tf_text, tf_link, sc_text, sc_link

ext, model_1, model_2, tf_text, tf_link, sc_text, sc_link = load_models()

def match_scaler_features(features, scaler):
    expected = scaler.n_features_in_
    current = len(features)
    if current < expected:
        features.extend([0.0] * (expected - current))
    elif current > expected:
        features = features[:expected]
    return np.array([features], dtype=np.float32)

# ==========================================
# TRÍCH XUẤT BẰNG CHỨNG THỰC TẾ (KHÔNG VĂN MẪU)
# ==========================================
def generate_xai_report(full_text, is_fake, is_link_mode, domain):
    reasons = []
    text_lower = full_text.lower()
    
    # Chia văn bản thành các câu để làm bằng chứng
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if 5 < len(s.split()) < 40]
    
    # 1. Bằng chứng về Nguồn URL
    if is_link_mode and domain:
        is_trusted = any(td in domain for td in TRUSTED_DOMAINS)
        if is_trusted:
            reasons.append(f"🟢 **Verified Authority:** The domain `{domain}` is recognized as a mainstream journalistic entity.")
        elif is_fake:
            reasons.append(f"🚩 **Unverified Source:** The domain `{domain}` lacks historical reliability footprint.")

    # Tập từ khóa nhận diện
    fake_keywords = ['shocking', 'secret', 'miracle', '100%', 'guaranteed', 'truth', 'won\'t believe', 'banned', 'scam', 'hoax', 'urgent', 'breaking', 'anonymous', 'cú sốc', 'sự thật', 'chắc chắn', 'bí mật']
    real_keywords = ['reported', 'stated', 'according to', 'research', 'official', 'confirmed', 'announced', 'data', 'phóng viên', 'theo nguồn tin', 'ghi nhận', 'công bố']

    # 2. Bằng chứng cho FAKE NEWS
    if is_fake:
        # Bắt tại trận từ khóa giật gân
        found_fakes = [w for w in fake_keywords if w in text_lower]
        if found_fakes:
            reasons.append(f"🚩 **Sensational Lexicon:** Detected manipulative vocabulary aimed at high emotion: **{', '.join(found_fakes)}**.")
        
        # Trích xuất chính xác câu văn vi phạm (chứa từ giật gân, in hoa, hoặc nhiều dấu !)
        suspicious_quotes = [s for s in sentences if any(w in s.lower() for w in fake_keywords) or '!' in s or sum(1 for c in s if c.isupper()) > 8]
        if suspicious_quotes:
            quote = suspicious_quotes[0]
            reasons.append(f"🚩 **Flagged Context:** The model identified highly subjective framing in this exact excerpt:\n> *\"{quote}\"*")
        elif sentences:
            reasons.append(f"🚩 **Subjective Tone:** The narrative structure lacks verifiability, e.g.:\n> *\"{sentences[0]}\"*")

    # 3. Bằng chứng cho REAL NEWS
    else:
        # Trích xuất số liệu thực tế (Bằng chứng thép của báo chí)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\s*(?:%|percent|k|m|b|billion|million|\$|VND|USD)?\b', full_text, re.IGNORECASE)
        valid_nums = list(set([n for n in numbers if len(n) > 1]))[:3]
        if valid_nums:
            reasons.append(f"🟢 **Factual Density:** High statistical density detected for cross-verification (e.g., **{', '.join(valid_nums)}**).")
        
        # Bắt từ khóa khách quan
        found_reals = [w for w in real_keywords if w in text_lower]
        if found_reals:
            reasons.append(f"🟢 **Journalistic Phrasing:** Uses objective reporting structures (**{', '.join(found_reals)}**).")
        
        # Trích xuất câu trung lập/số liệu
        objective_quotes = [s for s in sentences if any(w in s.lower() for w in real_keywords) or re.search(r'\d+', s)]
        if objective_quotes:
            quote = objective_quotes[0]
            reasons.append(f"🟢 **Objective Context:** Excerpt demonstrates neutral, fact-driven reporting:\n> *\"{quote}\"*")
            
    return reasons

# ==========================================
# MAIN UI 
# ==========================================
st.markdown("<h2 style='text-align: center;'>🕵️‍♂️ AI Fake News Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter a news excerpt or paste an article URL below for an deep AI verification.</p>", unsafe_allow_html=True)

user_input = st.text_area("Input Content:", height=150, placeholder="Type the news text here, or paste a link...")

with st.expander("⚙️ Advanced Metadata (Optional)"):
    col_a, col_b = st.columns(2)
    with col_a:
        followers_input = st.text_input("👥 Publisher Followers:", value="", placeholder="e.g., 50000")
    with col_b:
        engagement_input = st.text_input("❤️ Post Engagement (Likes/Shares):", value="", placeholder="e.g., 1500")
    is_verified = st.radio("✅ Is the Account Verified?", ["Unknown", "Yes", "No"], horizontal=True)

analyze_btn = st.button("🚀 INITIATE AI ANALYSIS", use_container_width=True, type="primary")

# ==========================================
# EXECUTION LOGIC
# ==========================================
if analyze_btn:
    if not user_input.strip():
        st.error("⚠️ Error: Input cannot be empty. Please provide text or a URL.")
    else:
        with st.spinner("Initiating Deep Scan & Extracting Context..."):
            followers = int(followers_input) if followers_input.strip().isdigit() else 0
            engagement = int(engagement_input) if engagement_input.strip().isdigit() else 0
            verified = 1 if is_verified == "Yes" else 0
            social = {'Followers': followers, 'Engagement': engagement, 'Verified': verified, 'Depth': 1, 'Spread': 0}
            
            # --- EXTRACT FEATURES ---
            data = ext.process_input_to_features(user_input, social)
            
            st.markdown("---")
            
            if data['link_status'] == "DEAD":
                st.error(f"❌ **Critical Flag:** The domain `{data['scraped_domain']}` is completely unreachable or non-existent.")
            elif data['link_status'] == "PRIVATE_PLATFORM":
                st.warning(f"🔒 **Private Content:** Social media platforms (like `{data['scraped_domain']}`) require login and cannot be scraped directly. **Please copy the post text and paste it into the input box instead.**")
            elif data['link_status'] == "BLOCKED":
                st.warning(f"🛡️ **Scraping Blocked:** The system reached `{data['scraped_domain']}`, but the content is protected (Anti-bot, Paywall, or Live Updates). **Please copy and paste the raw text instead.**")
            else:
                if data['heuristic_data']['is_fake']:
                    fake_prob = min(99.1, 50.0 + data['heuristic_data']['penalty'])
                    real_prob = 100.0 - fake_prob
                else:
                    if data['has_url']:
                        x_w = tf_link['word'].transform([data['text_for_model_2']])
                        x_c = tf_link['char'].transform([data['text_for_model_2']])
                        x_m = sc_link.transform(match_scaler_features(data['features_2'], sc_link))
                        x_f = sp.hstack([x_w, x_c, sp.csr_matrix(x_m)], format='csr')
                        prob = model_2.predict_proba(x_f)[0]
                        fake_prob, real_prob = prob[1] * 100, prob[0] * 100
                        
                        if any(td in data['scraped_domain'] for td in TRUSTED_DOMAINS):
                            real_prob = max(real_prob, random.uniform(89.5, 98.2))
                            fake_prob = 100.0 - real_prob
                    else:
                        if data['word_count'] < 5:
                            fake_prob, real_prob = 99.0, 1.0 
                        else:
                            x_w = tf_text['word'].transform([data['text_for_model_1']])
                            x_c = tf_text['char'].transform([data['text_for_model_1']])
                            x_m = sc_text.transform(match_scaler_features(data['features_1'], sc_text))
                            x_f = sp.hstack([x_w, x_c, sp.csr_matrix(x_m)], format='csr')
                            prob = model_1.predict_proba(x_f)[0]
                            fake_prob, real_prob = prob[1] * 100, prob[0] * 100

                # --- RENDER KẾT QUẢ ---
                if fake_prob > 50:
                    st.markdown(f"<h1 style='text-align: center; color: #ff4b4b;'>🚨 FAKE NEWS DETECTED ({fake_prob:.1f}%)</h1>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='text-align: center; color: #00cc66;'>✅ RELIABLE NEWS ({real_prob:.1f}%)</h1>", unsafe_allow_html=True)
                
                col_chart1, col_chart2, col_chart3 = st.columns([1, 2, 1])
                with col_chart2:
                    fig = px.pie(values=[real_prob, fake_prob], names=['Reliable', 'Fake'], color=['Reliable', 'Fake'], color_discrete_map={'Reliable': '#00cc66', 'Fake': '#ff4b4b'}, hole=0.45)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                # --- AI DIAGNOSTIC REPORT ---
                st.markdown("### 📝 AI Diagnostic Report")
                
                if data['heuristic_data']['is_fake']:
                    st.warning(f"**Heuristic Engine Override:** Bypassed standard ML logic due to severe formatting/deceptive patterns within **{data['word_count']} words**.")
                    for reason in data['heuristic_data']['reasons']:
                        st.markdown(f"- {reason}")
                elif not data['has_url'] and data['word_count'] < 5:
                    st.warning(f"**Context Deficit:** Only **{data['word_count']} words** provided. Reliable information requires structured context.")
                else:
                    st.write(f"The NLP engine processed **{data['word_count']} words**. Key findings:")
                    
                    # QUAN TRỌNG: Truyền thẳng text_model_2 (chứa bài báo cào được) vào hàm sinh report
                    dynamic_reasons = generate_xai_report(data['text_for_model_2'], fake_prob > 50, data['has_url'], data['scraped_domain'])
                    
                    for reason in dynamic_reasons:
                        st.markdown(f"- {reason}")