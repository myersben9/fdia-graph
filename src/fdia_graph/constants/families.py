"""The attack families: the code a file's `data/family` column carries for each, and the named subsets."""

from __future__ import annotations

# Code -> name: every code a released file can carry, the single-snapshot families of data releases
# v0.8.3 and earlier included, so old releases keep loading and scoring per family.
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al", 7: "Am"}
FAMILY_CODE = {name: code for code, name in FAMILIES.items()}  # name -> code: "Aq" -> 1
BENIGN_CODE = FAMILY_CODE["benign"]  # 0: a frame under no attack

AT = "At"  # the slow ramp, multi-snapshot [WU26]
AM = "Am"  # the overload attack, multi-snapshot [WU26]
AT_CODE = FAMILY_CODE[AT]  # 5
AM_CODE = FAMILY_CODE[AM]  # 7

# The codes of the families that are local false states passing the residual test.
STEALTHY_FAMILIES = {FAMILY_CODE[n] for n in ("Aq", AT, "Al", AM)}
# The codes of As and Ar, kept out of train and val in the unseen-attack protocol [BOY22].
HELDOUT_CODES = {FAMILY_CODE[n] for n in ("As", "Ar")}
# What the generator makes: the multi-snapshot families [WU26].
GENERATED_FAMILIES = (AT, AM)
# The families the New York release builds, by the names its files carry.
NEW_YORK_FAMILIES = (AM, AT)
