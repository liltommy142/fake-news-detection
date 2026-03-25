#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std;

// BFS
void calculateBFS(const json& node, int currentDepth, int& maxDepth, int& totalNodes) {
    totalNodes++;
    maxDepth = max(maxDepth, currentDepth);
    if (node.is_object() || node.is_array()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            calculateBFS(it.value(), currentDepth + 1, maxDepth, totalNodes);
        }
    }
}

// HEURISTIC
double getAdvancedScore(const string& text) {
    if (text.empty()) return 0;
    double score = 0;
    string low = text;
    transform(low.begin(), low.end(), low.begin(), ::tolower);

    // List of keywords with associated weights
    vector<pair<vector<string>, double>> megaDict = {
        {{"breaking", "urgent", "emergency", "alert", "shocking", "bombshell", "critical", "happening", "just in", "panic", "terror", "immediate", "warning", "deadly", "chaos", "drastic", "flare up", "explosion"}, 6.5},
        {{"exposed", "truth", "hidden", "conspiracy", "leaked", "blackout", "forbidden", "censored", "secret", "coverup", "scandal", "fraud", "mystery", "insider", "anonymous", "agenda", "classified", "prophecy", "prediction"}, 8.0},
        {{"fake", "hoax", "rumor", "unconfirmed", "allegedly", "claims", "suspicious", "misleading", "lie", "false", "unverified", "debunked", "misinformation", "propaganda", "manipulated", "scam", "fabricated", "phony"}, 9.0},
        {{"retweet", "share this", "spread", "must read", "must watch", "viral", "copy paste", "don't ignore", "rt please", "tell everyone", "broadcast", "repost", "fav", "plz share", "keep this moving", "circulation"}, 7.0},
        {{"attack", "shooting", "gunman", "arrested", "hostage", "killed", "dead", "explosion", "raid", "police", "military", "victim", "casualty", "suspect", "siege", "gunfire", "blood", "officer down", "fatal"}, 5.0}
    };

    for (auto& group : megaDict) {
        for (const string& k : group.first) {
            size_t pos = low.find(k);
            while (pos != string::npos) { score += group.second; pos = low.find(k, pos + 1); }
        }
    }

    // Heuristic dựa trên ký tự đặc biệt
    for (char c : text) if (c == '!' || c == '?') score += 1.8;
    
    // Đếm số lượng hashtag (#) - Đặc điểm nhận dạng tin đồn mạnh
    size_t hashtags = count(text.begin(), text.end(), '#');
    score += (hashtags * 3.0);

    stringstream ss(text); string w;
    while (ss >> w) {
        bool allUpper = true;
        for(char c : w) if(isalpha(c) && islower(c)) allUpper = false;
        if (allUpper && w.length() > 3) score += 4.5;
    }
    return score + (double)text.length() / 40.0;
}

// --- Format file ---
string smartFormat(string t, size_t width) {
    t = regex_replace(t, regex("http\\S+"), ""); // Xóa URL
    t = regex_replace(t, regex("[^\\x20-\\x7E]"), ""); // Chỉ giữ ký tự ASCII sạch
    t.erase(remove_if(t.begin(), t.end(), [](char c){ return c=='\n'||c=='\r'||c=='\t'||c=='|'; }), t.end());

    if (t.length() > width) {
        
        size_t cutPos = width - 4;
        
        
        size_t lastSpace = t.find_last_of(" ", cutPos);
        
       
        if (lastSpace != string::npos && lastSpace > width / 2) {
            t = t.substr(0, lastSpace) + "...";
        } else {
            t = t.substr(0, cutPos) + "...";
        }
    }
    
    
    if (t.length() < width) t.append(width - t.length(), ' ');
    return t;
}

int main() {
    string root = "./Pheme";
    if (!fs::exists(root)) root = "./pheme";
    
    cout << "Processing dataset, please wait..." << endl;
    ofstream out("input_for_model.csv");
    
    const int W_ID = 22, W_AUTH = 18, W_SC = 10, W_DP = 8, W_SP = 8, W_TXT = 70;

    out << left << setw(W_ID) << "Thread_ID" << " | " << setw(W_AUTH) << "Author" << " | "
        << setw(W_SC) << "Score" << " | " << setw(W_DP) << "Depth" << " | "
        << setw(W_SP) << "Spread" << " | " << setw(W_TXT) << "Content_Snippet" << " | Label" << endl;
    out << string(155, '-') << endl;

    for (const auto& event : fs::directory_iterator(root)) {
        if (!event.is_directory()) continue;
        for (const auto& type : fs::directory_iterator(event.path())) {
            string folder = type.path().filename().string();
            string label = (folder == "rumours") ? "fake" : (folder == "non-rumours" ? "real" : "");
            if (label == "") continue;

            for (const auto& thread : fs::directory_iterator(type.path())) {
                if (!thread.is_directory() || thread.path().filename().string().substr(0, 2) == "._") continue;

                fs::path sP = thread.path() / "structure.json";
                fs::path sD = thread.path() / "source-tweet";
                if (!fs::exists(sD)) sD = thread.path() / "source-tweets";

                if (fs::exists(sP) && fs::file_size(sP) > 0 && fs::exists(sD)) {
                    for (const auto& f : fs::directory_iterator(sD)) {
                        if (f.path().extension() == ".json" && fs::file_size(f.path()) > 0) {
                            ifstream fIn(f.path()); json j;
                            try {
                                fIn >> j;
                                string txt = j.value("text", "");
                                string auth = j.contains("user") ? j["user"].value("screen_name", "unk") : "unk";

                                ifstream fStr(sP); json jStr; fStr >> jStr;
                                int dp = 0, sp = 0;
                                for (auto it = jStr.begin(); it != jStr.end(); ++it) calculateBFS(it.value(), 1, dp, sp);

                                out << smartFormat(thread.path().filename().string(), W_ID) << " | "
                                    << smartFormat(auth, W_AUTH) << " | "
                                    << left << setw(W_SC) << fixed << setprecision(2) << getAdvancedScore(txt) << " | "
                                    << left << setw(W_DP) << dp << " | "
                                    << left << setw(W_SP) << sp << " | "
                                    << smartFormat(txt, W_TXT) << " | " << label << endl;
                                break;
                            } catch (...) { continue; }
                        }
                    }
                }
            }
        }
    }
    out.close();
    cout << "Processing completed!"<< endl;
    return 0;
}