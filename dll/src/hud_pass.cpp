// The HUD pass.
//
// Every screen-space draw arrives through the same gbh_* entry points, so the
// entry point says nothing about whether a quad is the score in the corner or
// a "$100" floating over a pedestrian. The difference matters on a wide screen:
// the first belongs against the edge of the display, the second has to stay
// over the pedestrian. What does say it is the call stack - GTA2 draws its whole
// HUD from one function (game_access.h, kHudDraw) - so the one call to that
// function is routed through a thunk that raises a flag for its duration.
//
// The parts. The HUD draw is nothing but nineteen calls in a row, one per part
// of the HUD (FUN_004CA440):
//
// Coordinates below are the 640x480 canvas, and the first value each routine
// computes is y, the second x.
//
//    0  FUN_004C6DA0  nothing drawn
//    1  FUN_004C9C20  score, lives, multiplier (from 639), weapon and ammo (638 less
//                     half the icon), power-ups leftward from it: right. The
//                     multiplayer score list (x 16) and timer: left
//    2  FUN_004C74A0  the marker over a target               - world
//    3  FUN_004C78A0  multiplayer names over the players     - world
//    4  FUN_004C7A30  a row of icons centred on 320           - centred
//    5  FUN_004C74F0  gang respect bars, x 16-93: left
//    6  FUN_004C7B70  health hearts from 551: right
//    7  FUN_004C9890  the zone name, framed, centred on 320   - centred
//    8  FUN_004C94F0  a box centred on 320                    - centred
//    9  FUN_004C9690  a line centred on 320                   - centred
//   10  FUN_004C9430  the talking head: portrait at x 32 near the bottom and
//                     the subtitle from x 64, stacked up from 480 - centred
//   11  FUN_004C92A0  counters in boxes near x 0: left
//   12  FUN_004C8C80  text items at positions of their own
//   13  FUN_004C84C0  arrows pointing at targets              - world
//   14  FUN_004C96F0  a line centred on 320                   - centred
//   15  FUN_004C8A40  the big messages                        - centred
//   16  FUN_004C9FA0  the pause and stats screen              - centred
//   17  FUN_004C8910  multiplayer chat from x 0: left
//   18  FUN_004C8710  the quit prompt, reached by a tail jump - centred
//
// Each call is pointed at a stub of its own that notes the part and jumps on to
// the function it replaced, so the overlay can keep parts apart: one part's
// element never decides where another's goes. That was the whole class of bug
// where the zone name, appearing between the corners, pulled the corners to the
// middle with it.

#include "hud_pass.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstring>

#include "game_access.h"
#include "log.h"

// File scope and plain C types, so the inline assembly below can name them.
static volatile uint8_t s_inHud = 0;
static volatile uint8_t s_part = 0xFF;
static uintptr_t s_hudDraw = game::kHudDraw;
static uintptr_t s_hudReturn = 0;

// Called in place of the HUD draw, with the game's this pointer still in ecx and
// its arguments still on the stack: take our own return address off, call the
// real function exactly as the game would have, and put it back. Whoever cleans
// the arguments - the callee's `ret n` or the caller - finds them where it
// expects. The HUD is drawn once a frame and never recursively, so one saved
// return address is enough.
static __declspec(naked) void HudDrawThunk() {
    __asm {
        mov byte ptr [s_inHud], 1
        mov byte ptr [s_part], 0xFF
        pop dword ptr [s_hudReturn]
        call dword ptr [s_hudDraw]
        mov byte ptr [s_inHud], 0
        mov byte ptr [s_part], 0xFF
        push dword ptr [s_hudReturn]
        ret
    }
}

// The stage-end arrow.
//
// The arrows part (13 above, FUN_004C84C0) walks the game's seventeen arrow slots, 0x7C
// bytes each, and draws each in use through FUN_004C82C0 (thiscall, ecx = the slot, no
// stack arguments). Every slot looks the same to the HUD, so what tells the arrow that
// leads out of the stage from the ones that lead to a job is how it was made.
//
// Each of the three single-player scripts (ste, wil, bil) holds exactly one script command
// 441 (0x1B9). The script machine runs it through FUN_00478E10, which allocates a slot,
// aims it at a map position (the target description at slot + 0x30 gets kind 1, so the
// dword at slot + 0x40 is 1) and sets the slot's type, the dword at slot + 0x20, to 5.
// That is the arrow that appears once the score target is met, and it sits behind the
// score checks in the script. Arrows aimed at a person, a car or a phone have kind 3 at
// slot + 0x40, and the only other arrow of type 5 is the multiplayer one (FUN_004C1FDC),
// which also aims at an object. So "type 5 and aimed at a position" is the stage-end
// arrow and nothing else.
//
// The two calls in the arrows part go to a thunk that returns at once for that slot while
// the switch is on, and otherwise jumps to the game's own routine.
static volatile uint8_t s_hideStageArrow = 0;
static uintptr_t s_arrowDraw = 0x004C82C0;

static __declspec(naked) void ArrowDrawThunk() {
    __asm {
        cmp byte ptr [s_hideStageArrow], 0
        je draw
        cmp dword ptr [ecx + 0x20], 5
        jne draw
        cmp dword ptr [ecx + 0x40], 1
        jne draw
        ret
    draw:
        jmp dword ptr [s_arrowDraw]
    }
}

// The stage-end message.
//
// Beside the arrow, the same part of each script puts up a message once the score target is
// met - "You did it! Follow the big pink marker to leave the Downtown District!", or "HEY! You
// now have enough cash to leave this District. Follow the marker...", or the Industrial "OK!"
// version - and the script goes round that part again after every job, so it comes back each
// time. They are text keys 1989, 2910 and 3752 (wil, ste and bil: each appears once, in a
// message command, opcode 279 or 118). Both opcodes reach one handler, FUN_004792B0, through the
// single jump at 0x4806C9; it reads the key from the command (the word at +8), prints it, and
// ends by handing the command to FUN_00476E50 (thiscall, ecx = the script thread, the command
// pushed, callee cleans) to move the thread on. With the switch on, for those three keys the
// thunk skips the printing and does only that last step. The "you don't have enough" messages
// (2911, 3753) are other keys and stay.
static uintptr_t s_msgHandler = 0x004792B0;
static uintptr_t s_msgAdvance = 0x00476E50;
static uintptr_t s_scriptCommandVar = 0x006644CC;   // address of the pointer to the running command

static __declspec(naked) void StageMessageThunk() {
    __asm {
        cmp byte ptr [s_hideStageArrow], 0
        je pass
        mov eax, dword ptr [s_scriptCommandVar]
        mov eax, dword ptr [eax]
        movzx eax, word ptr [eax + 8]
        cmp eax, 1989
        je swallow
        cmp eax, 2910
        je swallow
        cmp eax, 3752
        je swallow
    pass:
        jmp dword ptr [s_msgHandler]
    swallow:
        mov eax, dword ptr [s_scriptCommandVar]
        push dword ptr [eax]
        call dword ptr [s_msgAdvance]
        ret
    }
}

// The floating score numbers - the "$100" that bursts out of a pedestrian you run over or a car
// you hit.
//
// The game has one byte for them, at 0x005EAD5D, set at startup from the value exploding_on
// under ...\GTA2\Screen in the registry (FUN_004CB200 stores `exploding_on != 1`, the default
// being 1). Set means "no numbers": the level load (0x0045D184) then does not build the number
// manager, [0x0066F5DC], and the level end (0x0045D7FD) does not free it. Eight other places
// read the byte with `mov al, [0x005EAD5D]`: the manager's draw (0x0045A605) and update
// (0x0045C372), and six that start a number (0x004B8076, 0x004B8777, 0x004B88D1, 0x004B8B1F,
// 0x004B8BD3, 0x004B8C9C).
//
// Flipping the game's own byte while a level runs would be unsafe: switched off after a level
// loaded without a manager, the draw would call into nothing. So the build/free sites keep the
// game's byte, and those eight reads are pointed at a byte of ours instead. Each frame it is
// set to "the game's byte, or the user asked to hide them", so a number manager that exists is
// only ever switched off and on, and the registry value (and the game's own cheat that toggles
// the byte) still count.
//
// Starts at 1 so that nothing is drawn before the first frame has copied the game's byte over.
static volatile uint8_t s_scoreNumbersOff = 1;
static volatile uint8_t s_hideScoreNumbers = 0;

namespace {

int32_t RelTo(uintptr_t from, uintptr_t to);   // defined below

const uintptr_t kScoreNumberFlag = 0x005EAD5D;
const uintptr_t kScoreNumberReads[] = {0x0045A605, 0x0045C372, 0x004B8076, 0x004B8777,
                                       0x004B88D1, 0x004B8B1F, 0x004B8BD3, 0x004B8C9C};
bool g_scoreNumbersSwitch = false;

// Each site must be exactly `mov al, [0x005EAD5D]` (A0 5D AD 5E 00) before any is touched, so a
// different gta2.exe is left alone.
bool InstallScoreNumberSwitch() {
    for (uintptr_t at : kScoreNumberReads) {
        const uint8_t* site = reinterpret_cast<const uint8_t*>(at);
        uint32_t operand = 0;
        memcpy(&operand, site + 1, sizeof(operand));
        if (site[0] != 0xA0 || operand != static_cast<uint32_t>(kScoreNumberFlag)) return false;
    }
    const uint32_t ours = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&s_scoreNumbersOff));
    for (uintptr_t at : kScoreNumberReads) {
        uint8_t* site = reinterpret_cast<uint8_t*>(at);
        DWORD previous = 0;
        if (!VirtualProtect(site + 1, 4, PAGE_EXECUTE_READWRITE, &previous)) return false;
        memcpy(site + 1, &ours, sizeof(ours));
        VirtualProtect(site + 1, 4, previous, &previous);
    }
    return true;
}

const uintptr_t kArrowDrawCalls[] = {0x004C8514, 0x004C8524};
bool g_arrowHideInstalled = false;
const uintptr_t kMessageJump = 0x004806C9;   // jmp FUN_004792B0 in the script dispatcher
bool g_messageHideInstalled = false;

bool InstallMessageHide() {
    uint8_t* site = reinterpret_cast<uint8_t*>(kMessageJump);
    int32_t rel = 0;
    memcpy(&rel, site + 1, sizeof(rel));
    if (site[0] != 0xE9 || kMessageJump + 5 + static_cast<uintptr_t>(rel) != s_msgHandler) return false;
    DWORD previous = 0;
    if (!VirtualProtect(site, 5, PAGE_EXECUTE_READWRITE, &previous)) return false;
    const int32_t to = static_cast<int32_t>(reinterpret_cast<uintptr_t>(&StageMessageThunk) -
                                            (kMessageJump + 5));
    memcpy(site + 1, &to, sizeof(to));
    VirtualProtect(site, 5, previous, &previous);
    return true;
}

bool InstallArrowHide() {
    for (uintptr_t at : kArrowDrawCalls) {
        const uint8_t* site = reinterpret_cast<const uint8_t*>(at);
        int32_t rel = 0;
        memcpy(&rel, site + 1, sizeof(rel));
        if (site[0] != 0xE8 || at + 5 + static_cast<uintptr_t>(rel) != s_arrowDraw) return false;
    }
    for (uintptr_t at : kArrowDrawCalls) {
        uint8_t* site = reinterpret_cast<uint8_t*>(at);
        DWORD previous = 0;
        if (!VirtualProtect(site, 5, PAGE_EXECUTE_READWRITE, &previous)) return false;
        const int32_t to = RelTo(at, reinterpret_cast<uintptr_t>(&ArrowDrawThunk));
        memcpy(site + 1, &to, sizeof(to));
        VirtualProtect(site, 5, previous, &previous);
    }
    return true;
}

// The HUD draw's nineteen calls, in order: where each is, and where it goes. The
// last is a tail jump rather than a call.
struct PartCall {
    uintptr_t site;
    uintptr_t target;
    uint8_t opcode;   // 0xE8 call, 0xE9 jmp
};
const PartCall kParts[] = {
    {0x004CA450, 0x004C6DA0, 0xE8}, {0x004CA45B, 0x004C9C20, 0xE8},
    {0x004CA466, 0x004C74A0, 0xE8}, {0x004CA471, 0x004C78A0, 0xE8},
    {0x004CA47C, 0x004C7A30, 0xE8}, {0x004CA487, 0x004C74F0, 0xE8},
    {0x004CA492, 0x004C7B70, 0xE8}, {0x004CA49A, 0x004C9890, 0xE8},
    {0x004CA4A1, 0x004C94F0, 0xE8}, {0x004CA4AC, 0x004C9690, 0xE8},
    {0x004CA4B7, 0x004C9430, 0xE8}, {0x004CA4C2, 0x004C92A0, 0xE8},
    {0x004CA4CD, 0x004C8C80, 0xE8}, {0x004CA4D8, 0x004C84C0, 0xE8},
    {0x004CA4E3, 0x004C96F0, 0xE8}, {0x004CA4EE, 0x004C8A40, 0xE8},
    {0x004CA4F9, 0x004C9FA0, 0xE8}, {0x004CA504, 0x004C8910, 0xE8},
    {0x004CA510, 0x004C8710, 0xE9},
};
const int kPartCount = static_cast<int>(sizeof(kParts) / sizeof(kParts[0]));
bool g_partsKnown = false;

int32_t RelTo(uintptr_t from, uintptr_t to) { return static_cast<int32_t>(to - (from + 5)); }

// Points each of the HUD draw's calls at a stub of its own:
//   mov byte ptr [s_part], k     C6 05 <&s_part> k
//   jmp <original target>        E9 <rel32>
// Nothing else is touched: ecx, the stack and the return address are exactly
// what the call left them.
bool InstallParts() {
    for (const PartCall& p : kParts) {
        const uint8_t* site = reinterpret_cast<const uint8_t*>(p.site);
        int32_t rel = 0;
        memcpy(&rel, site + 1, sizeof(rel));
        if (site[0] != p.opcode || p.site + 5 + static_cast<uintptr_t>(rel) != p.target) return false;
    }
    const size_t kStub = 12;
    uint8_t* stubs = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, kStub * kPartCount, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!stubs) return false;
    const uint32_t partAddress = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&s_part));
    for (int k = 0; k < kPartCount; ++k) {
        uint8_t* s = stubs + kStub * k;
        s[0] = 0xC6;
        s[1] = 0x05;
        memcpy(s + 2, &partAddress, 4);
        s[6] = static_cast<uint8_t>(k);
        s[7] = 0xE9;
        const int32_t back = RelTo(reinterpret_cast<uintptr_t>(s + 7), kParts[k].target);
        memcpy(s + 8, &back, 4);
    }
    FlushInstructionCache(GetCurrentProcess(), stubs, kStub * kPartCount);
    for (int k = 0; k < kPartCount; ++k) {
        uint8_t* site = reinterpret_cast<uint8_t*>(kParts[k].site);
        DWORD previous = 0;
        if (!VirtualProtect(site, 5, PAGE_EXECUTE_READWRITE, &previous)) return false;
        const int32_t to = RelTo(kParts[k].site, reinterpret_cast<uintptr_t>(stubs + kStub * k));
        memcpy(site + 1, &to, 4);
        VirtualProtect(site, 5, previous, &previous);
    }
    return true;
}

}  // namespace

namespace gta2dx9 {

void HudPassInstall() {
    uint8_t* site = reinterpret_cast<uint8_t*>(game::kHudDrawCall);
    int32_t rel = 0;
    memcpy(&rel, site + 1, sizeof(rel));
    if (site[0] != 0xE8 || game::kHudDrawCall + 5 + static_cast<uintptr_t>(rel) != game::kHudDraw) {
        Log("hud pass: gta2.exe is not the build this knows; the HUD keeps its 4:3 placement");
        return;
    }
    uint8_t call[5] = {0xE8};
    const int32_t to = static_cast<int32_t>(reinterpret_cast<uintptr_t>(&HudDrawThunk) -
                                            (game::kHudDrawCall + 5));
    memcpy(call + 1, &to, sizeof(to));
    DWORD previous = 0;
    if (!VirtualProtect(site, sizeof(call), PAGE_EXECUTE_READWRITE, &previous)) {
        Log("hud pass: could not patch %08X", static_cast<unsigned>(game::kHudDrawCall));
        return;
    }
    memcpy(site, call, sizeof(call));
    VirtualProtect(site, sizeof(call), previous, &previous);
    g_partsKnown = InstallParts();
    g_arrowHideInstalled = InstallArrowHide();
    g_messageHideInstalled = InstallMessageHide();
    g_scoreNumbersSwitch = InstallScoreNumberSwitch();
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    Log("hud pass: installed (%s; stage-end arrow switch %s, its message switch %s, score-number switch %s)",
        g_partsKnown ? "parts told apart" : "parts not told apart - one HUD for placement",
        g_arrowHideInstalled ? "ready" : "unavailable",
        g_messageHideInstalled ? "ready" : "unavailable",
        g_scoreNumbersSwitch ? "ready" : "unavailable");
}

// The game's own byte, as set from the registry: nonzero means it builds no number manager, so the
// numbers cannot be turned on from here.
bool ScoreNumbersOffByGame() {
    return *reinterpret_cast<const volatile uint8_t*>(kScoreNumberFlag) != 0;
}
bool HideScoreNumbers() { return s_hideScoreNumbers != 0; }
void SetHideScoreNumbers(bool on) { s_hideScoreNumbers = on ? 1 : 0; }
bool HideScoreNumbersAvailable() { return g_scoreNumbersSwitch; }

void ScoreNumbersTick() {
    if (!g_scoreNumbersSwitch) return;
    s_scoreNumbersOff = (s_hideScoreNumbers || ScoreNumbersOffByGame()) ? 1 : 0;
}

bool HideStageArrow() { return s_hideStageArrow != 0; }
void SetHideStageArrow(bool on) { s_hideStageArrow = on ? 1 : 0; }
bool HideStageArrowAvailable() { return g_arrowHideInstalled; }

bool HudPassActive() { return s_inHud != 0; }

int HudPassPart() {
    if (!g_partsKnown || !s_inHud || s_part >= kPartCount) return -1;
    return s_part;
}

bool HudPartIsWorldAnchored(int part) { return part == 2 || part == 3 || part == 13; }

bool HudPartIsCentred(int part) {
    switch (part) {
        case 4: case 7: case 8: case 9: case 10: case 14: case 15: case 16: case 18:
            return true;
        default:
            return false;
    }
}

}  // namespace gta2dx9
