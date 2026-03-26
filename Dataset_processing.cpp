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
// MODULE 1: HỆ THỐNG ĐÁNH GIÁ NGỮ NGHĨA
// ======================================================================
double getHeuristicScore(const string& text) {
    if (text.empty()) return 0.00;
    
    double score = 0.5 + ((double)text.length() / 150.0);
    string low = text;
    transform(low.begin(), low.end(), low.begin(), ::tolower);

    static const string t1_fake[] = {"not a drill", "spread this", "conspiracy", "hoax", "cover up", "scam", "deep state",
                                     "sheeple", "media is hiding", "exposed", "truth about", "do your research", "fake news"};
    static const string t2_fake[] = {"is this real", "unbelievable", "big if true", "shocking", "anonymous source", "leaked",
                                     "wake up", "bombshell", "propaganda", "mind blowing", "rumor", "rumour", "secret"};
    static const string t1_real[] = {"confirmed by", "official statement", "police report", "verified", "press release", "debunked",
                                     "fact check", "authorities", "official sources", "nypd", "lapd", "bbc news", "reuters", "ap news"};
    static const string t2_real[] = {"according to", "statement from", "spokesperson", "investigation", "police department",
                                     "mayor", "announced", "briefing", "update"};

    for (const auto& w : t1_fake) if (low.find(w) != string::npos) score += 10.0;
    for (const auto& w : t2_fake) if (low.find(w) != string::npos) score += 5.0;
    for (const auto& w : t1_real) if (low.find(w) != string::npos) score -= 10.0;
    for (const auto& w : t2_real) if (low.find(w) != string::npos) score -= 5.0;

    int exclamation = 0, question = 0, upper = 0;
    for (char c : text) {
        if (c == '!') exclamation++;
        else if (c == '?') question++;
        else if (isupper(c)) upper++;
    }
    
    score += (exclamation * 2.0) + (question * 1.5);
    if (text.length() > 10 && (double)upper / text.length() > 0.4) score += 8.0;
    
    return score;
}

// ======================================================================
// MODULE 2: BFS LAN TRUYỀN
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
// MODULE 3: CẮT CHỮ AN TOÀN
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

// ======================================================================
// MAIN ENGINE - SMOOTH STREAMING MODE
// ======================================================================
int main() {
    // 1. Tối ưu luồng I/O của C++
    ios_base::sync_with_stdio(false); 
    cin.tie(NULL);
    
    string root = fs::exists("./Pheme") ? "./Pheme" : "./pheme";
    ofstream out("input_for_model.csv");
    
    // KHÔNG dùng pubsetbuf nữa. 
    // Mặc định C++ sẽ dùng buffer nội bộ cực kỳ tối ưu của hệ điều hành, giúp ghi file chảy mượt như nước.
    
    out << "Thread_ID                 | Author          | Score    | Depth  | Spread | Ver | Follow       | AccAge | Engage       | Src | Content_Snippet                                                                  | Label\n";
    out << string(205, '-') << "\n";

    cout << "\n[!] STATUS: EXTRACTING DATA..." << endl;
    
    int total_processed = 0;

    // 2. Tái sử dụng vùng nhớ. Khai báo 1 lần duy nhất ngoài vòng lặp!
    string fileContent;
    fileContent.reserve(512 * 1024); // Đặt sẵn 512KB để không bao giờ phải cấp phát lại RAM.
    ifstream fi; // Tái sử dụng 1 luồng đọc file duy nhất

    for (const auto& event : fs::directory_iterator(root)) {
        if (!event.is_directory()) continue;
        
        for (const auto& type : fs::directory_iterator(event.path())) {
            if (!type.is_directory()) continue;
            string label = (type.path().filename().string() == "rumours") ? "fake" : "real";

            for (const auto& thread : fs::directory_iterator(type.path())) {
                if (!thread.is_directory() || thread.path().filename().string()[0] == '.') continue;
                
                fs::path sP = thread.path() / "structure.json";
                int dp = 0, sp = 0;
                
                // Đọc file structure
                if (fs::exists(sP)) {
                    fi.open(sP, ios::binary | ios::ate);
                    if (fi.is_open()) {
                        streamsize size = fi.tellg();
                        fi.seekg(0, ios::beg);
                        fileContent.resize(size); // Chỉ đổi kích thước, không tạo vùng nhớ mới
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

                                        // \n giúp ghi vào buffer nhanh chóng, HĐH sẽ tự flush khi buffer đầy
                                        out << thread_id << " | "
                                            << fastFormat(auth, 15) << " | "
                                            << left << setw(8) << fixed << setprecision(2) << getHeuristicScore(txt) << " | "
                                            << left << setw(6) << dp << " | "
                                            << left << setw(6) << sp << " | "
                                            << left << setw(3) << ver << " | "
                                            << left << setw(12) << followers << " | "
                                            << left << setw(6) << age << " | "
                                            << left << setw(12) << fixed << setprecision(2) << engage << " | "
                                            << left << setw(3) << (isSource ? "1" : "0") << " | "
                                            << fastFormat(txt, 80) << " | "
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
    cout << "\n[!] SUCCESS: Finished processing " << total_processed << " samples.\n";
    return 0;
}