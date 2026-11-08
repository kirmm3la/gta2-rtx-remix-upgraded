#include "cache_warmup.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "log.h"

namespace gta2dx9 {

// Defined in live_geometry.cpp: the hand-made cars' glass and normal maps, made at full speed.
bool HdPreloadStart();
bool HdPreloadProgress(int* done, int* total);

namespace {

// What the pre-loader's bar reads. State 2 (finished) is also what it is when no thread was
// started, so a warm-up that is off never holds the bar.
std::atomic<int> g_warmState{2};   // 0 listing files, 1 reading, 2 finished
std::atomic<unsigned long long> g_warmBytesDone{0}, g_warmBytesTotal{0};
std::atomic<unsigned> g_warmFilesDone{0}, g_warmFilesTotal{0};
bool g_preload = true;             // [preload] enabled
bool g_preloadSkip = false;        // [preload] allow_skip
int g_preloadTimeout = 300;        // [preload] timeout_seconds

std::mutex g_mutex;
std::wstring g_gameDir;                     // with its closing backslash
std::vector<std::wstring> g_session;        // paths used this session, relative to the game folder
std::unordered_set<std::wstring> g_sessionSet;
int g_pruneDays = 10;                       // cache files unused this long are removed; 0 keeps everything
bool g_dirty = false;
volatile LONG g_started = 0;
int g_maxMegabytes = 2048;
constexpr size_t kMaxListEntries = 30000;

struct Entry {
    std::wstring path;
    int day = 0;   // the last day it was used, in days since 1970
};

int Today() { return static_cast<int>(time(nullptr) / 86400); }

std::wstring Lower(std::wstring s) {
    for (wchar_t& c : s) c = static_cast<wchar_t>(std::towlower(c));
    return s;
}

void FindGameDirectory() {
    if (!g_gameDir.empty()) return;
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir = exe;
    const size_t slash = dir.find_last_of(L"\\/");
    g_gameDir = slash == std::wstring::npos ? std::wstring() : dir.substr(0, slash + 1);
}

std::wstring ListPath() { return g_gameDir + L"gta2dx9_warm_list.txt"; }

// The list from earlier sessions, most recently used first: one "path|day" a line, under a
// "# since day" line saying when the list was started. Returns false when there is no list.
bool LoadList(std::vector<Entry>* out, int* since) {
    FILE* f = nullptr;
    *since = Today();
    if (_wfopen_s(&f, ListPath().c_str(), L"rb") != 0 || !f) return false;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
        if (!n) continue;
        if (line[0] == '#') {
            const int d = atoi(line + (strncmp(line, "# since ", 8) == 0 ? 8 : 1));
            if (d > 0) *since = d;
            continue;
        }
        Entry e;
        e.day = Today();
        const char* bar = strchr(line, '|');
        const size_t len = bar ? static_cast<size_t>(bar - line) : n;
        for (size_t i = 0; i < len; ++i) e.path.push_back(static_cast<wchar_t>(static_cast<unsigned char>(line[i])));
        if (bar && atoi(bar + 1) > 0) e.day = atoi(bar + 1);
        out->push_back(e);
    }
    fclose(f);
    return true;
}

// This session's files first, then the earlier list, cut to a sensible length. Written beside
// the old one and swapped in, so a game closed mid-write cannot leave half a list.
void SaveList(const std::vector<Entry>& previous, int since) {
    std::vector<Entry> all;
    const int today = Today();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        std::unordered_set<std::wstring> seen;
        for (auto it = g_session.rbegin(); it != g_session.rend(); ++it) {   // newest first
            if (seen.insert(Lower(*it)).second) all.push_back({*it, today});
        }
        for (const Entry& p : previous) {
            if (seen.insert(Lower(p.path)).second) all.push_back(p);
        }
        g_dirty = false;
    }
    if (all.size() > kMaxListEntries) all.resize(kMaxListEntries);
    const std::wstring path = ListPath();
    const std::wstring temp = path + L".tmp";
    FILE* f = nullptr;
    if (_wfopen_s(&f, temp.c_str(), L"wb") != 0 || !f) return;
    fprintf(f, "# since %d\r\n", since);
    for (const Entry& p : all) {
        std::string narrow;
        for (wchar_t c : p.path) narrow.push_back(c < 128 ? static_cast<char>(c) : '?');
        fprintf(f, "%s|%d\r\n", narrow.c_str(), p.day);
    }
    fclose(f);
    MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
}

// The cache folder grows for ever: every change to a setting makes new files and nothing
// ever removes the old ones. Anything not used for g_pruneDays is deleted; it is only a
// cache, the game makes what it needs again. It needs the list of files in use from one
// earlier session, so the first start after this was added only builds the list; the next
// one cleans. A file is kept if it was used, or written, within g_pruneDays.
void PruneCache(const std::vector<Entry>& previous, int since) {
    // Files from an older version of the mod that named them differently, beside the game
    // itself; nothing reads them any more.
    {
        size_t removed = 0;
        uint64_t freed = 0;
        WIN32_FIND_DATAW fd;
        HANDLE find = FindFirstFileW((g_gameDir + L"gta2dx9_cacheglass*.dds").c_str(), &fd);
        if (find != INVALID_HANDLE_VALUE) {
            do {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
                if (DeleteFileW((g_gameDir + fd.cFileName).c_str())) {
                    ++removed;
                    freed += (static_cast<uint64_t>(fd.nFileSizeHigh) << 32) | fd.nFileSizeLow;
                }
            } while (FindNextFileW(find, &fd));
            FindClose(find);
        }
        if (removed) Log("cache cleanup: removed %zu files (%.0f MB) left by an older version of the mod", removed, freed / 1048576.0);
    }
    if (g_pruneDays <= 0) return;
    const int today = Today();
    (void)since;
    if (previous.empty()) {
        Log("cache cleanup: waiting - there is no list of the files in use yet; play one session and it will clean up "
            "the next time the game starts");
        return;
    }
    std::map<std::wstring, int> lastUsed;
    for (const Entry& e : previous) lastUsed[Lower(e.path)] = e.day;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (const std::wstring& p : g_session) lastUsed[Lower(p)] = today;
    }
    const std::wstring dir = L"gta2dx9_cache\\";
    size_t kept = 0, removed = 0;
    uint64_t freed = 0;
    WIN32_FIND_DATAW fd;
    HANDLE find = FindFirstFileW((g_gameDir + dir + L"*").c_str(), &fd);
    if (find == INVALID_HANDLE_VALUE) return;
    FILETIME nowFt;
    GetSystemTimeAsFileTime(&nowFt);
    const uint64_t now = (static_cast<uint64_t>(nowFt.dwHighDateTime) << 32) | nowFt.dwLowDateTime;
    const uint64_t maxAge = static_cast<uint64_t>(g_pruneDays) * 86400ull * 10000000ull;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        const std::wstring name = fd.cFileName;
        if (name.size() < 5 || Lower(name.substr(name.size() - 4)) != L".dds") continue;
        const auto used = lastUsed.find(Lower(dir + name));
        if (used != lastUsed.end() && today - used->second < g_pruneDays) { ++kept; continue; }
        // Written recently, so not an old leftover whatever the list says.
        const uint64_t written = (static_cast<uint64_t>(fd.ftLastWriteTime.dwHighDateTime) << 32) | fd.ftLastWriteTime.dwLowDateTime;
        if (now > written && now - written < maxAge) { ++kept; continue; }
        if (DeleteFileW((g_gameDir + dir + name).c_str())) {
            ++removed;
            freed += (static_cast<uint64_t>(fd.nFileSizeHigh) << 32) | fd.nFileSizeLow;
        }
    } while (FindNextFileW(find, &fd));
    FindClose(find);
    Log("cache cleanup: removed %zu files not used for %d days (%.0f MB freed), kept %zu", removed, g_pruneDays,
        freed / 1048576.0, kept);
}

// Reads a whole file and throws it away: the point is only that the system keeps it.
// Returns the bytes read, 0 for a file that is not there.
uint64_t ReadThrough(const std::wstring& path, std::vector<uint8_t>& buffer) {
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) return 0;
    uint64_t total = 0;
    for (;;) {
        DWORD got = 0;
        if (!ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()), &got, nullptr) || got == 0) break;
        total += got;
        g_warmBytesDone += got;
    }
    CloseHandle(file);
    return total;
}

// The hand-made artwork: logos first (small, and on screen as soon as a gang car is), then the cars.
void ListArtwork(std::vector<std::wstring>* out) {
    const std::wstring dir = L"gta2dx9_hd\\";
    std::vector<std::wstring> logos, cars;
    WIN32_FIND_DATAW fd;
    HANDLE find = FindFirstFileW((g_gameDir + dir + L"*.dds").c_str(), &fd);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        const std::wstring name = fd.cFileName;
        (name.rfind(L"logo_", 0) == 0 ? logos : cars).push_back(dir + name);
    } while (FindNextFileW(find, &fd));
    FindClose(find);
    out->insert(out->end(), logos.begin(), logos.end());
    out->insert(out->end(), cars.begin(), cars.end());
}

DWORD WINAPI WarmThread(LPVOID) {
    // Lowest CPU and disk priority: the game's own loading goes first, always. Not with the
    // pre-loader: the game is held behind its bar, so there is nothing to give way to.
    if (!g_preload) {
        SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
        Sleep(2500);
    }

    std::vector<Entry> previous;
    int since = Today();
    LoadList(&previous, &since);
    std::vector<std::wstring> order;
    std::unordered_set<std::wstring> seen;
    for (const Entry& p : previous) {
        if (seen.insert(Lower(p.path)).second) order.push_back(p.path);
    }
    const size_t fromList = order.size();
    std::vector<std::wstring> art;
    ListArtwork(&art);
    for (const std::wstring& p : art) {
        if (seen.insert(Lower(p)).second) order.push_back(p);
    }
    if (previous.empty()) {
        Log("cache warm-up: no list from an earlier session yet (first run); reading the hand-made artwork only. "
            "The list is built as you play.");
    }

    const DWORD started = GetTickCount();
    const uint64_t budget = static_cast<uint64_t>(g_maxMegabytes) * 1024ull * 1024ull;
    std::vector<uint8_t> buffer(1 << 20);
    uint64_t bytes = 0;
    size_t files = 0, missing = 0;

    // How much there is to read, so the pre-loader's bar can say how far along it is: the sizes
    // of the files up to the budget, which is where the loop below stops.
    {
        uint64_t sum = 0;
        unsigned count = 0;
        for (const std::wstring& rel : order) {
            if (sum >= budget) break;
            WIN32_FILE_ATTRIBUTE_DATA info;
            if (!GetFileAttributesExW((g_gameDir + rel).c_str(), GetFileExInfoStandard, &info)) continue;
            sum += (static_cast<uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
            ++count;
        }
        g_warmBytesTotal = sum;
        g_warmFilesTotal = count;
        g_warmState = 1;
    }
    for (const std::wstring& rel : order) {
        if (bytes >= budget) break;
        const uint64_t got = ReadThrough(g_gameDir + rel, buffer);
        if (got) { bytes += got; ++files; ++g_warmFilesDone; } else { ++missing; }
    }
    g_warmState = 2;   // the bar can go; what follows (cleanup, the list) is for the background
    if (g_preload) SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
    Log("cache warm-up: read %zu files (%.0f MB) in %.1f s; %zu from the last session's list, %zu hand-made artwork, "
        "%zu no longer there",
        files, bytes / 1048576.0, (GetTickCount() - started) / 1000.0, fromList, art.size(), missing);

    PruneCache(previous, since);

    // From here on it only keeps the list up to date.
    for (;;) {
        Sleep(10000);
        bool dirty;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            dirty = g_dirty;
        }
        if (dirty) SaveList(previous, since);
    }
    return 0;
}

}  // namespace

void CacheWarmupNote(const std::wstring& path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    FindGameDirectory();
    std::wstring rel = path;
    if (!g_gameDir.empty() && rel.size() > g_gameDir.size() &&
        Lower(rel.substr(0, g_gameDir.size())) == Lower(g_gameDir)) {
        rel = rel.substr(g_gameDir.size());
    } else if (rel.find(L':') != std::wstring::npos) {
        return;   // somewhere else entirely
    }
    if (g_sessionSet.insert(Lower(rel)).second) {
        g_session.push_back(rel);
        g_dirty = true;
    }
}

void CacheWarmupStart() {
    if (InterlockedExchange(&g_started, 1)) return;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        FindGameDirectory();
    }
    const std::wstring ini = g_gameDir + L"gta2dx9_effects.ini";
    g_preload = GetPrivateProfileIntW(L"preload", L"enabled", 1, ini.c_str()) != 0;
    g_preloadSkip = GetPrivateProfileIntW(L"preload", L"allow_skip", 0, ini.c_str()) != 0;
    g_preloadTimeout = static_cast<int>(GetPrivateProfileIntW(L"preload", L"timeout_seconds", 300, ini.c_str()));
    if (g_preloadTimeout < 10) g_preloadTimeout = 10;
    if (GetPrivateProfileIntW(L"cache", L"warmup", 1, ini.c_str()) == 0) {
        Log("cache warm-up: off ([cache] warmup=0)");
        return;
    }
    g_warmState = 0;
    g_maxMegabytes = static_cast<int>(GetPrivateProfileIntW(L"cache", L"max_megabytes", 2048, ini.c_str()));
    if (g_maxMegabytes < 64) g_maxMegabytes = 64;
    g_pruneDays = static_cast<int>(GetPrivateProfileIntW(L"cache", L"prune_days", 10, ini.c_str()));
    HANDLE thread = CreateThread(nullptr, 0, WarmThread, nullptr, 0, nullptr);
    if (thread) CloseHandle(thread);
    else g_warmState = 2;   // no thread, nothing to wait for
}

// ---- The start-up pre-loader ---------------------------------------------------------------

namespace {
DWORD g_preloadStart = 0;
bool g_preloadBegun = false;
bool g_preloadHd = false;        // the cars' preparation was started and is worth waiting for
bool g_preloadFinished = false;
}  // namespace

bool PreloadWanted() { return g_preload && !g_preloadFinished; }

WarmupProgress CacheWarmupProgress() {
    WarmupProgress p;
    p.finished = g_warmState.load() == 2;
    p.bytesDone = g_warmBytesDone.load();
    p.bytesTotal = g_warmBytesTotal.load();
    p.filesDone = g_warmFilesDone.load();
    p.filesTotal = g_warmFilesTotal.load();
    return p;
}

void PreloadBegin() {
    if (g_preloadBegun) return;
    g_preloadBegun = true;
    g_preloadStart = GetTickCount();
    g_preloadHd = HdPreloadStart();
    Log("preload: the game is held at its first frame until the hand-made cars are prepared and the cache is read "
        "(cars %s, cache %s)",
        g_preloadHd ? "yes" : "off", g_warmState.load() == 2 ? "off or empty" : "yes");
}

PreloadStatus PreloadPoll() {
    PreloadStatus s;
    s.done = false;
    const WarmupProgress w = CacheWarmupProgress();
    int hdDone = 0, hdTotal = 0;
    const bool hdActive = g_preloadHd && HdPreloadProgress(&hdDone, &hdTotal) && hdTotal > 0;
    const bool hdFinished = !hdActive || hdDone >= hdTotal;
    float hdFrac = hdActive ? static_cast<float>(hdDone) / hdTotal : 1.0f;
    float warmFrac = 1.0f;
    if (!w.finished) {
        warmFrac = w.bytesTotal ? static_cast<float>(static_cast<double>(w.bytesDone) / w.bytesTotal) : 0.0f;
        if (warmFrac > 1.0f) warmFrac = 1.0f;
    }
    // The cache is the long part when there is a lot of it, and the cars when there is not, so the
    // bar weighs each by how much there is to do rather than giving each half.
    float warmWeight = 0.5f + static_cast<float>(w.bytesTotal / (512.0 * 1048576.0));
    if (warmWeight > 4.0f) warmWeight = 4.0f;
    s.fraction = (hdFrac + warmWeight * warmFrac) / (1.0f + warmWeight);
    if (s.fraction > 1.0f) s.fraction = 1.0f;

    _snprintf(s.title, sizeof(s.title) - 1, "Getting the HD cars ready");
    if (!w.finished && w.bytesTotal == 0) {
        _snprintf(s.detail, sizeof(s.detail) - 1, "Looking through the picture cache...");
    } else {
        _snprintf(s.detail, sizeof(s.detail) - 1, "Cars %d of %d   -   pictures %u of %u files (%llu of %llu MB)",
                  hdDone, hdTotal, w.filesDone, w.filesTotal, w.bytesDone / 1048576ull, w.bytesTotal / 1048576ull);
    }
    _snprintf(s.hint, sizeof(s.hint) - 1, "%s",
              g_preloadSkip ? "Press Esc to skip. The game starts when the bar is full."
                            : "The game starts when the bar is full. It only takes long the first time.");

    bool skip = false;
    if (g_preloadSkip && (GetAsyncKeyState(VK_ESCAPE) & 0x8000)) {
        DWORD pid = 0;
        if (HWND foreground = GetForegroundWindow()) GetWindowThreadProcessId(foreground, &pid);
        skip = pid == GetCurrentProcessId();
    }
    const bool timedOut = static_cast<int>((GetTickCount() - g_preloadStart) / 1000) >= g_preloadTimeout;
    if ((hdFinished && w.finished) || skip || timedOut) {
        s.done = true;
        s.fraction = 1.0f;
        g_preloadFinished = true;
        Log("preload: %s after %.1f s (cars %d/%d, cache %u files, %llu MB)",
            skip ? "skipped" : (timedOut ? "gave up waiting" : "done"), (GetTickCount() - g_preloadStart) / 1000.0,
            hdDone, hdTotal, w.filesDone, w.bytesDone / 1048576ull);
    }
    s.title[sizeof(s.title) - 1] = s.detail[sizeof(s.detail) - 1] = s.hint[sizeof(s.hint) - 1] = '\0';
    return s;
}

}  // namespace gta2dx9
