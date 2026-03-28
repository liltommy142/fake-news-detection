//                       _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//           	Protected by the Buddha 
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <deque>
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std;

// ======================================================================
// MODULE 1: CẤU TRÚC ĐẶC TRƯNG NGÔN NGỮ (NLP FEATURES)
// ======================================================================
struct NLPFeatures {
    double negative_score = 0.0;
    double positive_score = 0.0;
    double clickbait_score = 0.0;
    double caps_ratio = 0.0;
    double punct_density = 0.0;
    double avg_word_len = 0.0;
};

// ======================================================================
// MODULE 2: HỆ THỐNG TỪ ĐIỂN TỐI ƯU BỘ NHỚ (CACHE-FRIENDLY)
// ======================================================================
static const vector<string> fake_lexicon = {
    "shocking", "unbelievable", "not a drill", "spread this", "conspiracy",
    "hoax", "cover up", "scam", "deep state", "sheeple", "media is hiding",
    "exposed", "truth about", "do your research", "fake news", "is this real",
    "big if true", "anonymous source", "leaked", "wake up", "bombshell",
    "propaganda", "mind blowing", "rumor", "rumour", "secret", "confidential",
    "must see", "miracle", "urgent", "censored", "false flag", "mainstream media", 
    "click here", "you won't believe", "viral",
    "hot", "drama", "scandal", "full clip", "uncensored", "leaked clip",
    "netizen", "netizens", "idol", "showbiz", "sugar baby", "sugar daddy", 
    "hot girl", "hot boy", "vip", "link in bio", "inbox", "ib", "share gấp"
};

static const vector<string> real_lexicon = {
    "confirmed by", "official statement", "police report", "verified",
    "press release", "debunked", "fact check", "authorities", "official sources",
    "bbc news", "reuters", "ap news", "according to", "statement from",
    "spokesperson", "investigation", "police department", "mayor", "announced",
    "briefing", "update", "testimony", "evidence", "court ruling", "witnesses", 
    "reported by", "academic study", "scientific", "expert says", "press conference",
    "vtv", "vnexpress", "tuoi tre", "thanh nien", "ministry of", "health department",
    "public security", "official page", "press agency", "government portal"
};

// ======================================================================
// MODULE 3: TRÍCH XUẤT NGỮ NGHĨA (FEATURE ENGINEERING ĐỒNG ĐỀU)
// ======================================================================
NLPFeatures extractNLPFeatures(const string& text) {
    NLPFeatures feat;
    if (text.empty()) return feat;

    int text_len = text.length();
    int upper_count = 0, punct_count = 0, char_count = 0, word_count = 0;
    bool in_word = false;
    
    string low;
    low.reserve(text_len);

    for (char c : text) {
        low += tolower(c); 
        if (isupper(c)) upper_count++;
        if (c == '!' || c == '?' || c == '*') punct_count++;
        
        if (isalpha(c)) {
            char_count++;
            if (!in_word) { word_count++; in_word = true; }
        } else {
            in_word = false;
        }
    }

    feat.caps_ratio = (double)upper_count / text_len * 100.0;
    feat.punct_density = (double)punct_count / text_len * 100.0;
    if (word_count > 0) feat.avg_word_len = (double)char_count / word_count;

    for (const string& word : fake_lexicon) {
        if (low.find(word) != string::npos) feat.clickbait_score += 1.0;
    }
    for (const string& word : real_lexicon) {
        if (low.find(word) != string::npos) feat.positive_score += 1.0;
    }

    if (feat.caps_ratio > 15.0) feat.clickbait_score *= 1.5; 
    if (feat.punct_density > 5.0) feat.clickbait_score *= 1.2; 

    if (feat.caps_ratio < 5.0 && feat.punct_density < 2.0 && feat.avg_word_len >= 4.5) {
        feat.positive_score *= 1.5; 
    }

    feat.negative_score = (feat.clickbait_score * 2.0) + (feat.punct_density * 0.5) + (feat.caps_ratio * 0.2);
    return feat;
}

// ======================================================================
// MODULE 4: BFS LAN TRUYỀN
// ======================================================================
void calculateBFS(const json& structure, int& dp, int& sp) {
    dp = 0; sp = 0;
    if (!structure.is_object() || structure.empty()) return;
    
    deque<pair<const json*, int>> q;
    for (auto& [k, v] : structure.items()) q.push_back({&v, 1});
    
    while (!q.empty()) {
        auto [node, d] = q.front(); q.pop_front();
        sp++;
        if (d > dp) dp = d;
        if (node->is_object()) {
            for (auto& [k, v] : node->items()) q.push_back({&v, d + 1});
        }
    }
}

// ======================================================================
// MODULE 5: CẮT CHỮ VÀ LÀM SẠCH ĐỊNH DẠNG
// ======================================================================
string fastFormat(const string& text, size_t width) {
    string out;
    out.reserve(width + 10); 
    bool in_space = false;
    
    for (size_t i = 0; i < text.length() && out.length() <= width + 5; ++i) {
        if (text.compare(i, 4, "http") == 0) {
            while (i < text.length() && text[i] != ' ') i++;
            continue;
        }
        char c = text[i];
        if (c >= 32 && c <= 126 && c != '|') {
            if (c == ' ') {
                if (!in_space && !out.empty()) { out += ' '; in_space = true; }
            } else {
                out += c; in_space = false;
            }
        }
    }
    
    if (!out.empty() && out.back() == ' ') out.pop_back();
    
    if (out.length() > width) {
        size_t cut_pos = width - 4; 
        while (cut_pos > 0 && out[cut_pos] != ' ') cut_pos--; 
        if (cut_pos == 0) cut_pos = width - 4; 
        out = out.substr(0, cut_pos) + "...";
    }
    
    out.resize(width, ' ');
    return out;
}

string cleanTextForCSV(const string& text) {
    string out;
    out.reserve(text.length()); 
    bool in_space = false;
    
    for (size_t i = 0; i < text.length(); ++i) {
        if (text.compare(i, 4, "http") == 0) {
            while (i < text.length() && text[i] != ' ') i++;
            continue;
        }
        
        char c = text[i];
        if (c == '\n' || c == '\r' || c == '|' || c == '\t') {
            c = ' ';
        }
        
        if (c >= 32 && c <= 126) {
            if (c == ' ') {
                if (!in_space && !out.empty()) { out += ' '; in_space = true; }
            } else {
                out += c; in_space = false;
            }
        }
    }
    
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ======================================================================
// MAIN ENGINE
// ======================================================================
int main() {
    ios_base::sync_with_stdio(false); 
    cin.tie(NULL);
    
    string root = fs::exists("./Pheme") ? "./Pheme" : "./pheme";
    if (!fs::exists(root)) {
        cout << "[!] Error: Data folder 'Pheme' not found in current directory!" << '\n';
        return 1;
    }

    ofstream out("input_for_model.csv");

    cout << "\n[!] STATUS: EXTRACTING DATA..." << '\n';

    // CĂN CHỈNH HEADER MỚI (Viết đầy đủ tên, đã đo khoảng cách)
    out << left 
        << setw(25) << "Thread_ID" << " | "
        << setw(15) << "Author" << " | "
        << setw(15) << "Negative_Score" << " | "
        << setw(15) << "Positive_Score" << " | "
        << setw(16) << "Clickbait_Score" << " | "
        << setw(12) << "Caps_Ratio" << " | "
        << setw(20) << "Punctuation_Density" << " | "
        << setw(16) << "Avg_Word_Length" << " | "
        << setw(6)  << "Depth" << " | "
        << setw(7)  << "Spread" << " | "
        << setw(9)  << "Verified" << " | "
        << setw(12) << "Followers" << " | "
        << setw(12) << "Account_Age" << " | "
        << setw(12) << "Engagement" << " | "
        << setw(10) << "Is_Source" << " | "
        << "Content_Snippet" << " | " 
        << "Label\n";
    out << string(235, '-') << "\n";
    
    int total_processed = 0;
    string fileContent;
    fileContent.reserve(512 * 1024); 
    ifstream fi; 

    for (const auto& event : fs::directory_iterator(root)) {
        if (!event.is_directory()) continue;
        
        for (const auto& type : fs::directory_iterator(event.path())) {
            if (!type.is_directory()) continue;
            string label = (type.path().filename().string() == "rumours") ? "fake" : "real";

            for (const auto& thread : fs::directory_iterator(type.path())) {
                if (!thread.is_directory() || thread.path().filename().string()[0] == '.') continue;
                
                fs::path sP = thread.path() / "structure.json";
                int dp = 0, sp = 0;
                
                if (fs::exists(sP)) {
                    fi.open(sP, ios::binary | ios::ate);
                    if (fi.is_open()) {
                        streamsize size = fi.tellg();
                        fi.seekg(0, ios::beg);
                        fileContent.resize(size); 
                        if (fi.read(&fileContent[0], size)) {
                            try {
                                json jS = json::parse(fileContent, nullptr, false);
                                if (!jS.is_discarded()) calculateBFS(jS, dp, sp);
                            } catch(...) {}
                        }
                        fi.close();
                    }
                }

                vector<string> folders = {"source-tweet", "source-tweets", "reactions"};
                int reaction_count = 0; 
                string thread_id = fastFormat(thread.path().filename().string(), 25); 

                for (const string& fName : folders) {
                    fs::path dirPath = thread.path() / fName;
                    if (!fs::exists(dirPath)) continue;

                    bool isSource = (fName.find("source") != string::npos);

                    for (const auto& f : fs::directory_iterator(dirPath)) {
                        if (f.path().extension() != ".json") continue; 

                        fi.open(f.path(), ios::binary | ios::ate);
                        if (!fi.is_open()) continue;
                        
                        streamsize size = fi.tellg();
                        fi.seekg(0, ios::beg);
                        fileContent.resize(size);
                        
                        if (fi.read(&fileContent[0], size)) {
                            try {
                                json j = json::parse(fileContent, nullptr, false);
                                if (!j.is_discarded()) {
                                    string txt = j.value("text", "");
                                    int followers = j["user"].value("followers_count", 0);
                                    bool isQualityReaction = (!isSource && (followers > 50 || txt.length() > 40));

                                    if (isSource || (isQualityReaction && reaction_count < 3)) {
                                        if (!isSource) reaction_count++; 

                                        string auth = j["user"].value("screen_name", "unk");
                                        int ver = j["user"].value("verified", false) ? 1 : 0;
                                        int friends = max(1, j["user"].value("friends_count", 1));
                                        double engage = (double)followers / friends;
                                        
                                        string cAt = j["user"].value("created_at", "");
                                        int age = (cAt.length() >= 4) ? (2026 - stoi(cAt.substr(cAt.length() - 4))) : 0;

                                        NLPFeatures nlp = extractNLPFeatures(txt);
                                        string cleaned_content = cleanTextForCSV(txt); 

                                        // CĂN CHỈNH DỮ LIỆU KHỚP VỚI HEADER
                                        out << left
                                            << setw(25) << thread_id << " | "
                                            << setw(15) << fastFormat(auth, 15) << " | "
                                            << setw(15) << fixed << setprecision(2) << nlp.negative_score << " | "
                                            << setw(15) << fixed << setprecision(2) << nlp.positive_score << " | "
                                            << setw(16) << fixed << setprecision(2) << nlp.clickbait_score << " | "
                                            << setw(12) << fixed << setprecision(2) << nlp.caps_ratio << " | "
                                            << setw(20) << fixed << setprecision(2) << nlp.punct_density << " | "
                                            << setw(16) << fixed << setprecision(2) << nlp.avg_word_len << " | "
                                            << setw(6)  << dp << " | "
                                            << setw(7)  << sp << " | "
                                            << setw(9)  << ver << " | "
                                            << setw(12) << followers << " | "
                                            << setw(12) << age << " | "
                                            << setw(12) << fixed << setprecision(2) << engage << " | "
                                            << setw(10) << (isSource ? "1" : "0") << " | "
                                            << cleaned_content << " | " 
                                            << label << "\n";
                                        
                                        total_processed++;
                                    }
                                }
                            } catch(...) {}
                        }
                        fi.close();
                    }
                }
            }
        }
    }
    
    out.close();
    cout << "\n[!] SUCCESS: Finished processing.\n";
    return 0;
}