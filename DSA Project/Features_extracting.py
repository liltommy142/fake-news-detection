import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import trafilatura

class RealtimeFeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def extract_urls(self, text):
        url_pattern = r'(?i)\b(?:(?:https?://|www\.)[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b'
        return [url.rstrip('.,;!"\'()[]{}') for url in re.findall(url_pattern, text)]

    def _get_nlp_meta(self, text):
        text = str(text)
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        caps_ratio = sum(1 for c in text if c.isupper()) / (char_count + 1e-5)
        excl_density = text.count('!') / (char_count + 1e-5)
        ques_density = text.count('?') / (char_count + 1e-5)
        punct_density = len(re.findall(r'[.,;:]', text)) / (char_count + 1e-5)
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
        
        return [word_count, char_count, caps_ratio, excl_density, ques_density, punct_density, avg_word_len]

    def process_input_to_features(self, user_input, social_params):
        # 1. Cảm xúc (Sentiment)
        source_sent = self.analyzer.polarity_scores(user_input)
        s_neg, s_pos = source_sent['neg'], source_sent['pos']
        s_gap = s_pos - s_neg
        clickbait = 0.5 if (user_input.count('!') > 2 or user_input.isupper()) else 0.1

        # 2. Xử lý Link
        source_urls = self.extract_urls(user_input)
        text_model_2 = user_input
        scraped_content = ""
        
        if source_urls:
            downloaded = trafilatura.fetch_url(source_urls[0])
            if downloaded:
                scraped_content = trafilatura.extract(downloaded) or ""
                text_model_2 += f" [SOURCE_LINK_CONTENT] {scraped_content}"

        nlp_features = self._get_nlp_meta(user_input)

        # 3. Gom Model 1 (17 cột gốc)
        features_1 = [
            s_neg, s_pos, clickbait, s_gap,           
            *nlp_features,                            
            0, 0, 0, 0,  # React
            0, 0         # Divergence
        ]

        # 4. Gom Model 2 (17 cột gốc + 7 cột mạng xã hội)
        features_2 = features_1 + [
            social_params.get('Depth', 1),
            social_params.get('Spread', 0),
            social_params.get('Verified', 0.5), # Default trung lập
            social_params.get('Followers', 50000),
            social_params.get('Engagement', 500),
            len(source_urls),
            0 
        ]

        return {
            'text_for_model_1': user_input,
            'text_for_model_2': text_model_2,
            'features_1': features_1,
            'features_2': features_2
        }