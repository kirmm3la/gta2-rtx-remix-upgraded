// Reads the mod's own files into the operating system's file cache ahead of need.
//
// After a restart nothing the mod uses is in memory yet: the first time each car turns up
// its hand-made maps and the recoloured artwork made for its paint are read from disk, and
// on a cold disk that is a stall on the frame. This runs on a thread of its own, at the
// lowest priority, and reads the files that were used last time (and the hand-made
// artwork) so they are already in memory when the game asks. It keeps a short list of the
// files each session used, gta2dx9_warm_list.txt, so the warm-up follows what you play.
//
// Settings, in the [cache] section of gta2dx9_effects.ini:
//   warmup=1            0 turns it off
//   max_megabytes=2048  the most it reads
//   prune_days=10       cache files not used for this many days are deleted (0 keeps everything)
//
// The cache folder otherwise only grows: a changed setting makes new files and the old
// ones are never touched again. After one session has built the list of files in use,
// the next start deletes the ones neither used nor written for prune_days.
#pragma once

#include <string>

namespace gta2dx9 {

// Starts the thread. Call once, after the game's window and the ini files are known.
void CacheWarmupStart();

// A file the mod just used from the cache or the artwork folder, so the next session can
// read it ahead. Cheap, safe from any thread, and meant for the moment a material is made,
// not for every frame.
void CacheWarmupNote(const std::wstring& path);

// ---- The start-up pre-loader ---------------------------------------------------------------
//
// With the pre-loader on (the default) the game is held at its first frame behind a full-screen
// bar until the hand-made cars are prepared and the cache files are read into memory, instead of
// that happening piece by piece while you play. The warm-up above then runs at full speed rather
// than at the lowest priority, since nothing else is going on.
//
// Settings, in the [preload] section of gta2dx9_effects.ini:
//   enabled=1           0 goes back to doing it all quietly in the background
//   allow_skip=0        1 lets Esc end the wait (the work carries on in the background)
//   timeout_seconds=300 the longest the bar holds the game, whatever has not finished
//
// What the bar shows, for the world view to draw.
struct PreloadStatus {
    bool done = true;
    float fraction = 1.0f;     // 0..1
    char title[64] = "";
    char detail[160] = "";
    char hint[96] = "";
};

// Whether the bar is to be shown at all this run.
bool PreloadWanted();
// Starts the cars' preparation at full speed and the clock for the time-out. Once.
void PreloadBegin();
// Where things stand. done turns true when everything is ready, the time-out has passed, or
// (allow_skip=1) Esc was pressed.
PreloadStatus PreloadPoll();

// What the warm-up has read so far, for the bar.
struct WarmupProgress {
    bool finished = true;   // nothing left to read, or the warm-up is off
    unsigned long long bytesDone = 0, bytesTotal = 0;
    unsigned filesDone = 0, filesTotal = 0;
};
WarmupProgress CacheWarmupProgress();

}  // namespace gta2dx9
