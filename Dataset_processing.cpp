#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <deque>
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std;

// --- HEURISTIC 6.0: KHÁCH QUAN HÓA & LOẠI BỎ ĐIỂM 0 ---
double getUltraHeuristic(const string& text) {
    if (text.empty()) return 0.50; // Trả về điểm tối thiểu thay vì 0
    
    // 1. ĐIỂM CƠ SỞ (Baseline): Dựa trên độ dài văn bản để tránh điểm 0 tuyệt đối
    double score = (double)text.length() / 25.0; 
    
    string low = text;
    transform(low.begin(), low.end(), low.begin(), ::tolower);

    // 2. NHÓM TỪ KHÓA NGHI VẤN CAO (Trọng số 6.0 - 9.0)
    static const vector<pair<string, double>> highRisk = {
        {"not a drill", 9.0}, {"is this real", 8.5}, {"government hide", 9.5},
        {"unbelievable", 7.0}, {"big if true", 8.0}, {"leaked footage", 9.0},
        {"anonymous source", 7.5}, {"spread this", 6.0}, {"what they won't tell", 9.5},
        {"shocking", 6.5}, {"terrorist", 5.0}, {"hostage", 5.0}, {"explosion", 5.5}
    };

    // 3. NHÓM TỪ KHÓA TRUNG TÍNH/BÁO CÁO (Để tweet có điểm nhưng không bị coi là fake)
    static const vector<pair<string, double>> neutralPhrases = {
        {"reporting", 3.0}, {"breaking news", 4.0}, {"update", 2.5},
        {"witnessed", 3.5}, {"seen at", 3.0}, {"happening now", 3.5},
        {"video shows", 4.0}, {"people are", 2.0}, {"police at", 3.5}
    };

    // 4. KHÁCH QUAN HÓA: Điểm trừ cho các nguồn xác thực (Penalty)
    static const vector<pair<string, double>> trustFactors = {
        {"confirmed by", -7.0}, {"official statement", -8.0}, {"bbc news", -6.0},
        {"reuters", -6.0}, {"ap news", -6.0}, {"police report", -5.0},
        {"verified account", -4.0}, {"sources say", -2.0}
    };

    for (auto& p : highRisk) if (low.find(p.first) != string::npos) score += p.second;
    for (auto& n : neutralPhrases) if (low.find(n.first) != string::npos) score += n.second;
    for (auto& t : trustFactors) if (low.find(t.first) != string::npos) score += t.second;

    // 5. Thưởng điểm cho việc sử dụng Hashtag và CapsLock (Dấu hiệu lan truyền)
    score += (count(text.begin(), text.end(), '#') * 2.5);
    int upper = count_if(text.begin(), text.end(), ::isupper);
    if (text.length() > 10 && (double)upper / text.length() > 0.3) score += 4.5;

    // Đảm bảo điểm số luôn nằm trong khoảng hợp lý, tối thiểu là 1.0 nếu có chữ
    return max(1.0, score); 
}

// --- BFS TỐI ƯU TỐC ĐỘ: DEQUE & MEMORY OPTIMIZATION ---
void calculateBFSSpeed(const json& structure, int& dp, int& sp) {
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

// --- SMART FORMAT (Chống lệch CSV) ---
string smartFormat(string t, size_t width) {
    t = regex_replace(t, regex("http\\S+|[^\\x20-\\x7E]"), ""); 
    t.erase(remove_if(t.begin(), t.end(), [](char c){ return c=='\n'||c=='\r'||c=='\t'||c=='|'; }), t.end());
    if (t.length() > width) {
        size_t lastSpc = t.find_last_of(" ", width - 4);
        t = (lastSpc != string::npos && lastSpc > width/2) ? t.substr(0, lastSpc) : t.substr(0, width - 4);
        t += "...";
    }
    return t.append(max((int)0, (int)(width - t.length())), ' ');
}

int main() {
    string root = fs::exists("./Pheme") ? "./Pheme" : "./pheme";
    ofstream out("input_for_model.csv");
    
    const int W_ID = 20, W_AUTH = 15, W_SC = 8, W_DP = 6, W_SP = 6, W_TXT = 60;
    out << "Thread_ID           | Author          | Score    | Depth  | Spread | Content_Snippet                                              | Label" << endl;
    out << string(140, '-') << endl;

    cout << "\n[!] STATUS: RUNNING OPTIMIZED ENGINE..." << endl;

    int total = 0;
    for (const auto& event : fs::directory_iterator(root)) {
        if (!event.is_directory()) continue;
        for (const auto& type : fs::directory_iterator(event.path())) {
            if (!type.is_directory()) continue;
            string folder = type.path().filename().string();
            string label = (folder == "rumours") ? "fake" : "real";
            if (folder.find("rumours") == string::npos) continue;

            for (const auto& thread : fs::directory_iterator(type.path())) {
                if (!thread.is_directory() || thread.path().filename().string()[0] == '.') continue;
                
                fs::path sP = thread.path() / "structure.json", sD = thread.path() / "source-tweet";
                if (!fs::exists(sD)) sD = thread.path() / "source-tweets";

                if (fs::exists(sP) && fs::exists(sD)) {
                    ifstream fsP(sP); json jS; try { fsP >> jS; } catch(...) { continue; }
                    int dp, sp; calculateBFSSpeed(jS, dp, sp);

                    for (const auto& f : fs::directory_iterator(sD)) {
                        if (f.path().extension() == ".json") {
                            ifstream fi(f.path()); json j;
                            try {
                                fi >> j;
                                string txt = j.value("text", ""), auth = j["user"].value("screen_name", "unk");
                                out << smartFormat(thread.path().filename().string(), W_ID) << " | "
                                    << smartFormat(auth, W_AUTH) << " | "
                                    << left << setw(W_SC) << fixed << setprecision(2) << getUltraHeuristic(txt) << " | "
                                    << left << setw(W_DP) << dp << " | "
                                    << left << setw(W_SP) << sp << " | "
                                    << smartFormat(txt, W_TXT) << " | " << label << endl;
                                total++;
                                break;
                            } catch(...) { continue; }
                        }
                    }
                }
            }
        }
    }
    out.close();
    cout << "[!] SUCCESS: Successfully processing dataset.\n" << endl;
    return 0;
}
