// Knowing when GTA2 is drawing its HUD, and which part of it, so the overlay can
// place the HUD on a wide screen without moving anything tied to the world. See
// hud_pass.cpp.
#pragma once

namespace gta2dx9 {

// Routes the game's one call to its HUD draw through us, and each of the HUD
// draw's own calls - one per part of the HUD - as well. Leaves a gta2.exe that is
// not the build these addresses came from alone.
void HudPassInstall();

// True while the game is inside its HUD draw.
bool HudPassActive();

// Which part of the HUD is drawing: the index of the call inside the HUD draw
// (FUN_004CA440) that is running, 0-18, or -1 outside one or when the parts
// could not be told apart.
int HudPassPart();

// Parts that draw at a place projected from the world - a marker over its
// target, a player's name over the player, the arrows pointing at a target -
// and so have to stay where the world is on any screen.
bool HudPartIsWorldAnchored(int part);

// Parts the game lays out around the middle of the screen - the zone name, the
// talking head and its subtitle, the big messages, the pause screen - which stay
// centred however little of them there is. A two-word subtitle beside its
// portrait sits nearest the left edge, and is still centred.
bool HudPartIsCentred(int part);

// The arrow that points at the way out of the stage, which the script puts up once the score
// target is met. With this on it is not drawn; every other arrow - the ones that lead to a
// job - is. Available only when the game is the build the addresses came from.
bool HideStageArrow();
void SetHideStageArrow(bool on);
bool HideStageArrowAvailable();

// The floating score numbers that burst out of a pedestrian you run over or a car you hit. With this
// on they are not drawn (the game's own registry value exploding_on=0 does the same, and wins: it
// builds no number manager at all, so they cannot be switched back on without restarting).
bool HideScoreNumbers();
void SetHideScoreNumbers(bool on);
bool HideScoreNumbersAvailable();
bool ScoreNumbersOffByGame();
// Once a frame: copies the game's own switch, or ours, to the byte the game now reads.
void ScoreNumbersTick();

}  // namespace gta2dx9
