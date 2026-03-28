import re
import random
import requests
from collections import Counter
import numpy as np
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import trafilatura
from urllib.parse import urlparse
import warnings

warnings.filterwarnings('ignore')

STOPWORDS = set([
    "the", "and", "to", "of", "a", "in", "that", "is", "for", "on", "it", "with", "as", "was",
    "this", "are", "be", "by", "an", "or", "from", "at", "which", "but", "not", "have", "has", "had"
])

class RealtimeFeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def extract_urls(self, text):
        if not isinstance(text, str): return []
        url_pattern = r'(?i)\b(?:(?:https?://|www\.)[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b'
        raw_urls = re.findall(url_pattern, text)
        
        clean_urls = []
        for url in raw_urls:
            url = url.rstrip('.,;!"\'()[]{}')
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            clean_urls.append(url)
        return list(set(clean_urls))

    def _get_nlp_meta(self, text):
        text = str(text).strip()
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        if char_count == 0:
            return [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]

        caps_ratio = sum(1 for c in text if c.isupper()) / (char_count + 1e-5)
        excl_density = text.count('!') / (char_count + 1e-5)
        ques_density = text.count('?') / (char_count + 1e-5)
        punct_density = len(re.findall(r'[.,;:]', text)) / (char_count + 1e-5)
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
        
        return [word_count, char_count, caps_ratio, excl_density, ques_density, punct_density, avg_word_len]

    def advanced_heuristic_check(self, full_text):
        text_lower = full_text.lower()
        penalty_score = 0
        reasons = []

        # 1. TỪ KHÓA ĐÁNH LỪA NHẬN THỨC & PR CÔNG NGHỆ BẨN (TECH-SCAM)
        vague_authorities = ["leading international institute", "experts say", "researchers claim", "scientists have recently announced", "early reports claim", "anonymous sources", "a startup company has claimed"]
        
        # Thêm các từ khóa bắt thóp "sạc 5 giây", "chữa bách bệnh"
        hyperboles = ["revolutionary", "breakthrough", "detailed life forecast", "within seconds", "transform education", "miracle cure", "100% guaranteed", "mind-blowing", "in just", "entire month on a single", "game changer", "defies physics", "never before seen"]
        
        clickbaits = ["you won't believe", "before it's deleted", "what happens next", "doctors hate him", "viral sensation", "act now", "don't ignore this", "shocking truth"]

        found_vague = [phrase for phrase in vague_authorities if phrase in text_lower]
        found_hyperbole = [phrase for phrase in hyperboles if phrase in text_lower]
        found_clickbait = [phrase for phrase in clickbaits if phrase in text_lower]

        if len(found_vague) >= 1:
            penalty_score += 30 * len(found_vague)
            reasons.append(f"🚩 **Vague Sources:** The text relies heavily on unverified citations ({', '.join(found_vague)}).")
        
        if len(found_hyperbole) >= 1:
            # Tăng điểm phạt cực nặng cho các cụm từ cường điệu phi thực tế
            penalty_score += 45 * len(found_hyperbole)
            reasons.append(f"🚩 **Exaggerated Claims:** Detected hyperbolic or scientifically improbable vocabulary ({', '.join(found_hyperbole)}).")
        
        if len(found_clickbait) >= 1:
            penalty_score += 45
            reasons.append(f"🚩 **Clickbait Framing:** Uses manipulative emotional triggers ({', '.join(found_clickbait)}).")

        # 2. BẮT LỖI LOGIC SỐ LIỆU (REGEX NÂNG CAO)
        # Bắt các pattern kiểu "in just 5 seconds... for an entire month"
        if re.search(r'(in just|only)\s+\d+\s+(seconds|minutes)', text_lower) and re.search(r'(entire month|years|forever)', text_lower):
            penalty_score += 60  # Phạt thẳng tay vượt mốc 50
            reasons.append("🚩 **Logical Fallacy:** Claims contain highly improbable time/performance ratios (e.g., 'seconds' vs 'months').")

        # 3. ĐỊNH DẠNG & NGỮ PHÁP
        words = full_text.split()
        if len(words) > 0:
            caps_words = [w for w in words if w.isupper() and len(re.sub(r'\W', '', w)) > 2]
            caps_ratio = len(caps_words) / len(words)
            if caps_ratio > 0.15 and len(caps_words) >= 3:
                penalty_score += 40
                reasons.append(f"🚩 **Formatting Anomaly:** Excessive use of ALL CAPS ({len(caps_words)} words).")

            excessive_punc = re.findall(r'[!?]{2,}', full_text)
            if len(excessive_punc) >= 1 or full_text.count('!') >= 3:
                penalty_score += 40
                reasons.append("🚩 **Punctuation Abuse:** Detected highly emotional or unprofessional punctuation.")

        is_fake_flag = True if penalty_score >= 50 else False
        return is_fake_flag, min(penalty_score, 100), reasons

    def extract_xai_evidence(self, text):
        words = text.split()
        clean_words = [re.sub(r'\W+', '', w).lower() for w in words]
        meaningful_words = [w for w in clean_words if len(w) > 4 and w not in STOPWORDS]
        top_keywords = [word for word, count in Counter(meaningful_words).most_common(3)]
        
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\s*(?:%|percent|k|m|b|billion|million|\$|VND|USD|EUR)?\b', text, re.IGNORECASE)
        unique_numbers = list(set([n for n in numbers if len(n) > 1]))[:3]
        
        sentences = re.split(r'[.!?]\s+', text)
        valid_sentences = [s.strip() for s in sentences if 8 <= len(s.split()) <= 35]
        snippet = random.choice(valid_sentences) if valid_sentences else ""
        
        return top_keywords, unique_numbers, snippet

    def process_input_to_features(self, user_input, social_params):
        # 1. Phân tích Cảm xúc Nguồn (Sentiment)
        source_sent = self.analyzer.polarity_scores(user_input)
        s_neg, s_pos, s_neu, s_comp = source_sent['neg'], source_sent['pos'], source_sent['neu'], source_sent['compound']
        s_gap = s_pos - s_neg
        clickbait = 0.5 if (user_input.count('!') > 2 or user_input.isupper()) else 0.1

        # 2. Xử lý Link
        source_urls = self.extract_urls(user_input)
        text_model_2 = user_input
        scraped_content = ""
        link_status = "OK"
        scraped_domain = ""
        
        if source_urls:
            target_url = source_urls[0]
            scraped_domain = urlparse(target_url).netloc
            private_domains = ['facebook.com', 'instagram.com', 'tiktok.com', 'linkedin.com', 'x.com', 'twitter.com']
            
            if any(domain in scraped_domain.lower() for domain in private_domains):
                link_status = "PRIVATE_PLATFORM"
            else:
                downloaded = trafilatura.fetch_url(target_url)
                if downloaded:
                    scraped_content = trafilatura.extract(downloaded) or ""
                
                if len(scraped_content.split()) < 15:
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        res = requests.get(target_url, headers=headers, timeout=5)
                        if res.status_code == 200:
                            soup = BeautifulSoup(res.text, 'html.parser')
                            paragraphs = soup.find_all('p')
                            scraped_content = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
                    except Exception:
                        pass
                
                if not scraped_content:
                    link_status = "DEAD"
                elif len(scraped_content.split()) < 15:
                    link_status = "BLOCKED"
                else:
                    text_model_2 += f" [SOURCE_LINK_CONTENT] {scraped_content} "

        # 3. Chạy Heuristics & XAI Data
        nlp_features = self._get_nlp_meta(user_input)
        is_adv_fake, heuristic_penalty, heuristic_reasons = self.advanced_heuristic_check(text_model_2)
        keywords, numbers, snippet = self.extract_xai_evidence(text_model_2)

        features_1 = [
            s_neg, s_pos, clickbait, s_gap,           
            *nlp_features,                            
            0, 0, 0, 0,  
            0, 0         
        ]

        features_2 = features_1 + [
            social_params.get('Depth', 1),
            social_params.get('Spread', 0),
            social_params.get('Verified', 0.5), 
            social_params.get('Followers', 50000),
            social_params.get('Engagement', 500),
            len(source_urls),
            0  
        ]

        return {
            'text_for_model_1': user_input,
            'text_for_model_2': text_model_2,
            'features_1': features_1,
            'features_2': features_2,
            'has_url': len(source_urls) > 0,
            'link_status': link_status,
            'scraped_domain': scraped_domain,
            'word_count': len(text_model_2.split()),
            'heuristic_data': {'is_fake': is_adv_fake, 'penalty': heuristic_penalty, 'reasons': heuristic_reasons},
            'xai_data': {
                'keywords': keywords, 
                'numbers': numbers, 
                'snippet': snippet,
                # TRÍCH XUẤT THÊM THÁI ĐỘ CẢM XÚC ĐỂ APP.PY SỬ DỤNG
                'sentiment': {'pos': s_pos, 'neg': s_neg, 'neu': s_neu, 'compound': s_comp} 
            }
        }