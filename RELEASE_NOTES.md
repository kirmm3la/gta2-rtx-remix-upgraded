# GTA2 RTX Upgraded 1.3

A Direct3D 9 renderer for GTA2 that lets RTX Remix path-trace the game, built on
[gebdag/gta2-rtx-remix](https://github.com/gebdag/gta2-rtx-remix) (MIT), now updated to gebdag's newest version, v0.8.5. This release adds a hand-made visual layer on top.

## Install

1. **Install RTX Remix:** download `Remix_Plus_v1.5.1_x86_games_release.zip` from the **Assets** list at the bottom of the
   [Remix Plus 1.5.1 release page](https://github.com/RemixProjGroup/dxvk-remix/releases/tag/remix-plus-1.5.1) and unzip it into your
   GTA2 folder next to `gta2.exe`. (Without it the game runs, but without ray tracing.)
2. **Unzip this release** into the same GTA2 folder. Say yes to replacing files.
3. **Start the game.** **F4** (or the game's own **Options** button) opens the settings panel.

The zip contains no game files, no RTX Remix binaries and no player profiles, and installing it never touches your saves.
The mod can autosave after a big payout; to turn that off, untick **Save after a big payout** in the F4 menu and click Save,
or add `[save]` and `auto_save=0` to `gta2dx9_settings.ini`.

## Where to get GTA2

GTA2 was officially free for a while: Rockstar gave it away as part of its **Rockstar Classics** series, together with the first
GTA and Wild Metal Country. That free offer has ended. Rockstar later sold it on its own store (until 2013, according to
[Delisted Games](https://delistedgames.com/?p=10001)), and Steam removed its GTA2 page in July 2025 without explanation.

So today there is no official place to get it. The game is still easy to find online if you look for it (searching for
"GTA 2 Rockstar Classics" is a good start), but those copies come from third parties, so use one you trust, or a copy you
already own (the original CD, or an old Rockstar or Steam purchase). This project ships no game files and does not host or
link to downloads of the game.

## What's new in 1.3

- **Updated to gebdag's newest version:** this release is built on the latest gta2-rtx-remix by gebdag (v0.8.5), so it includes
  everything new from the original project, such as its WidescreenFix and Plugin SDK support.
- **Start-up bar:** the game now starts with a progress bar that prepares the HD cars and loads the saved car pictures first,
  so nothing is built while you play. Turn it off with `enabled=0` under `[preload]` in `gta2dx9_effects.ini`.
- **New option:** F4 menu > "Score numbers" > "Hide the exploding score numbers" turns off the cash numbers that burst out when
  you run over a pedestrian or hit a car (the setting the original GTA2 Manager used to have). Press Save to keep it.

## What was new in 1.2

- **Smoother on a fresh install:** the first time a car model or a new paint colour appears, the mod used to stop the game for a
  moment to build its HD glass maps, door pictures and dents. That work now runs in the background, so new cars and doors no
  longer cause stalls. The first run on a new PC still has to build its cache, but it no longer freezes the game while doing so.
- **Fixed:** the Rednecks emblem now appears only on the dark-blue Rednecks pickup, on its cab roof, not on every pickup in
  stage 2.

## What was new in 1.1

- **Fixed:** the car radio no longer goes silent when you drive away from its source. It now stays in stereo and fades out
  with distance, as intended.
- **Safer exit helper:** the small helper that closes a game that hangs on exit now runs Windows' own `cmd.exe`, `ping.exe` and
  `taskkill.exe` by their full system paths, so a look-alike file in the game folder can never be run instead.
- **Autosave documented:** the README now explains the autosave after big payouts and how to turn it off.

## What's in it

- Hand-made HD cars with full PBR maps (albedo, normal, roughness, metal) and real glass
- Doors and dents drawn as shading on the HD cars; wheels, roof domes, gang roof logos
- Taxi, police and emergency lights; blood puddles on the road
- Radio stays in stereo at any distance (original distance effect is a menu option)
- Voice packs for gangs, cops and pedestrians
- Emissive maps and animated water as Remix mods

## Known issues

- Tank turret still uses the game's original low-resolution art and looks out of place next to the HD hull.
- 5-star FBI cars show blue instead of black.
- Some cars still have unmarked windows or false glass; a few sprites (pink sports car, limo) are still low-res.
- Daytime light trails (denoiser), pedestrian flicker and dusk reflections need more work.

## Credits and licence

MIT. See `gta2dx9_LICENSE.txt` and `gta2dx9_THIRD_PARTY.txt` (Dear ImGui, NVIDIA RTX Remix API headers).
GTA2 is a trademark and copyright of Rockstar Games / Take-Two. Unofficial fan project.
