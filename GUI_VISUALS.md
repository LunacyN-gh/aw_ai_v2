# Board graphics and playback

Restart the GUI to load the new vector terrain and unit graphics. Blue and Red
have distinct colors, acted units are muted, and unit abbreviations plus HP
remain visible. Sprites scale with the board; units render above terrain.

The **Animation speed** dropdown controls human and AI move/combat playback:
0.25×, 0.5×, 1×, 2×, 4×, 8×, or Instant. Thinking time remains a separate setting.
At 1×, movement takes about 0.18 seconds per traversed tile and both combatants
flash for about 0.5 seconds. Speed changes apply from the next action. Damage
and destruction appear after the flash; gameplay commits each action once.

Movement follows a shortest legal terrain-cost path, including detours around
enemy blockers and impassable tiles. Friendly units may be crossed but not
occupied, matching the current rules. These paths are presentation only and do
not change search or game mechanics. Pause after turn retains its existing
behavior. Restart/load invalidate pending playback callbacks.
