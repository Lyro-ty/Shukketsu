"""Item database with curated TBC Rogue gear, enchants, gems, and set bonuses.

Provides an in-memory lookup for ~50 key items spanning all gear slots and
phases 1-5, used by the simulation engine for gear resolution and stat totaling.
"""

from code.shukketsu.sim.models import (
    Enchant,
    GearSlot,
    Gem,
    GemSlot,
    Item,
    ProcEffect,
    ProcTrigger,
    RogueSpec,
    SetBonus,
    WeaponStats,
    WeaponType,
)

# ---------------------------------------------------------------------------
# Curated items — ~50 key TBC Rogue items with real WoW item IDs
# ---------------------------------------------------------------------------

_CURATED_ITEMS: list[Item] = [
    # === Weapons ===
    # -- Main Hand --
    Item(
        id=28573,
        name="Blinkstrike",
        slot=GearSlot.MAIN_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 17, "stamina": 10},
        weapon=WeaponStats(min_damage=105, max_damage=196, speed=2.6, weapon_type=WeaponType.SWORD, dps=57.9),
        proc=ProcEffect(trigger=ProcTrigger.ON_HIT, rate=0.03, duration=0.0, effect={"teleport": 1.0}),
    ),
    Item(
        id=28189,
        name="Latro's Shifting Sword",
        slot=GearSlot.MAIN_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 23, "hit_rating": 15},
        weapon=WeaponStats(min_damage=115, max_damage=215, speed=2.7, weapon_type=WeaponType.SWORD, dps=61.1),
    ),
    Item(
        id=32837,
        name="Warglaive of Azzinoth MH",
        slot=GearSlot.MAIN_HAND,
        item_level=156,
        phase=3,
        stats={"agility": 22, "stamina": 31, "hit_rating": 21, "haste_rating": 21},
        weapon=WeaponStats(min_damage=214, max_damage=398, speed=2.8, weapon_type=WeaponType.SWORD, dps=109.3),
    ),
    Item(
        id=28311,
        name="Blade of Infamy",
        slot=GearSlot.MAIN_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 18, "stamina": 16, "crit_rating": 22},
        weapon=WeaponStats(min_damage=128, max_damage=193, speed=2.6, weapon_type=WeaponType.SWORD, dps=61.7),
    ),
    Item(
        id=30082,
        name="Talon of Azshara",
        slot=GearSlot.MAIN_HAND,
        item_level=128,
        phase=2,
        stats={"agility": 20, "stamina": 18, "hit_rating": 17, "expertise_rating": 14},
        weapon=WeaponStats(min_damage=130, max_damage=195, speed=2.6, weapon_type=WeaponType.FIST, dps=62.5),
    ),
    Item(
        id=28524,
        name="Emerald Ripper",
        slot=GearSlot.MAIN_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 22, "stamina": 13, "crit_rating": 18},
        weapon=WeaponStats(min_damage=72, max_damage=134, speed=1.8, weapon_type=WeaponType.DAGGER, dps=57.2),
    ),
    Item(
        id=34164,
        name="Mounting Vengeance",
        slot=GearSlot.MAIN_HAND,
        item_level=159,
        phase=5,
        stats={"agility": 24, "stamina": 24, "crit_rating": 23, "expertise_rating": 20},
        weapon=WeaponStats(min_damage=222, max_damage=413, speed=2.8, weapon_type=WeaponType.SWORD, dps=113.4),
    ),
    # -- Off Hand --
    Item(
        id=32838,
        name="Warglaive of Azzinoth OH",
        slot=GearSlot.OFF_HAND,
        item_level=156,
        phase=3,
        stats={"agility": 22, "stamina": 31, "hit_rating": 21, "haste_rating": 21},
        weapon=WeaponStats(min_damage=107, max_damage=199, speed=1.4, weapon_type=WeaponType.SWORD, dps=109.3),
    ),
    Item(
        id=32471,
        name="Shard of Azzinoth",
        slot=GearSlot.OFF_HAND,
        item_level=141,
        phase=3,
        stats={"agility": 20, "stamina": 22, "crit_rating": 14},
        weapon=WeaponStats(min_damage=97, max_damage=181, speed=1.5, weapon_type=WeaponType.DAGGER, dps=92.7),
    ),
    Item(
        id=28295,
        name="Gladiator's Shiv",
        slot=GearSlot.OFF_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 15, "stamina": 22, "crit_rating": 14, "hit_rating": 10},
        weapon=WeaponStats(min_damage=65, max_damage=99, speed=1.4, weapon_type=WeaponType.DAGGER, dps=58.6),
    ),
    Item(
        id=34346,
        name="Hammer of Judgement",
        slot=GearSlot.OFF_HAND,
        item_level=154,
        phase=5,
        stats={"agility": 18, "stamina": 24, "haste_rating": 24},
        weapon=WeaponStats(min_damage=138, max_damage=258, speed=1.5, weapon_type=WeaponType.MACE, dps=132.0),
    ),
    # === Trinkets ===
    Item(
        id=28830,
        name="Dragonspine Trophy",
        slot=GearSlot.TRINKET_1,
        item_level=115,
        phase=1,
        proc=ProcEffect(
            trigger=ProcTrigger.PPM,
            rate=1.0,
            icd=20.0,
            duration=10.0,
            effect={"haste_rating": 325.0},
        ),
    ),
    Item(
        id=30627,
        name="Tsunami Talisman",
        slot=GearSlot.TRINKET_1,
        item_level=128,
        phase=2,
        proc=ProcEffect(
            trigger=ProcTrigger.ON_CRIT,
            rate=0.10,
            icd=45.0,
            duration=10.0,
            effect={"attack_power": 340.0},
        ),
    ),
    Item(
        id=32505,
        name="Madness of the Betrayer",
        slot=GearSlot.TRINKET_1,
        item_level=141,
        phase=3,
        stats={"crit_rating": 20},
        proc=ProcEffect(
            trigger=ProcTrigger.PPM,
            rate=1.0,
            icd=10.0,
            duration=10.0,
            effect={"armor_penetration": 300.0},
        ),
    ),
    Item(
        id=33831,
        name="Berserker's Call",
        slot=GearSlot.TRINKET_1,
        item_level=154,
        phase=5,
        stats={"attack_power": 90},
        on_use=ProcEffect(
            trigger=ProcTrigger.ON_USE,
            rate=1.0,
            icd=120.0,
            duration=20.0,
            effect={"attack_power": 360.0},
        ),
    ),
    Item(
        id=28163,
        name="Brooch of Deftness",
        slot=GearSlot.TRINKET_2,
        item_level=110,
        phase=1,
        stats={"hit_rating": 28, "expertise_rating": 14},
    ),
    Item(
        id=29383,
        name="Bloodlust Brooch",
        slot=GearSlot.TRINKET_2,
        item_level=115,
        phase=1,
        stats={"attack_power": 72},
        on_use=ProcEffect(
            trigger=ProcTrigger.ON_USE,
            rate=1.0,
            icd=120.0,
            duration=20.0,
            effect={"attack_power": 278.0},
        ),
    ),
    # === Head ===
    Item(
        id=29044,
        name="Netherblade Facemask",
        slot=GearSlot.HEAD,
        item_level=120,
        phase=1,
        stats={"agility": 40, "stamina": 30, "crit_rating": 28, "hit_rating": 18},
        sockets=[GemSlot.META, GemSlot.RED],
        socket_bonus={"agility": 4},
        set_id="netherblade",
    ),
    Item(
        id=31027,
        name="Slayer's Helm",
        slot=GearSlot.HEAD,
        item_level=146,
        phase=3,
        stats={"agility": 48, "stamina": 39, "crit_rating": 32, "hit_rating": 22, "haste_rating": 18},
        sockets=[GemSlot.META, GemSlot.RED],
        socket_bonus={"agility": 4},
        set_id="slayer",
    ),
    Item(
        id=34244,
        name="Duplicitous Guise",
        slot=GearSlot.HEAD,
        item_level=154,
        phase=5,
        stats={"agility": 52, "stamina": 43, "crit_rating": 30, "expertise_rating": 26, "haste_rating": 20},
        sockets=[GemSlot.META, GemSlot.YELLOW],
        socket_bonus={"hit_rating": 4},
    ),
    # === Neck ===
    Item(
        id=28762,
        name="Adornment of Stolen Souls",
        slot=GearSlot.NECK,
        item_level=115,
        phase=1,
        stats={"agility": 22, "stamina": 15, "crit_rating": 16, "hit_rating": 10},
    ),
    Item(
        id=30017,
        name="Telonicus's Pendant of Mayhem",
        slot=GearSlot.NECK,
        item_level=128,
        phase=2,
        stats={"agility": 26, "stamina": 18, "crit_rating": 22, "hit_rating": 12},
    ),
    Item(
        id=34177,
        name="Clutch of Demise",
        slot=GearSlot.NECK,
        item_level=154,
        phase=5,
        stats={"agility": 30, "stamina": 24, "crit_rating": 24, "haste_rating": 18},
    ),
    # === Shoulder ===
    Item(
        id=29048,
        name="Netherblade Pauldrons",
        slot=GearSlot.SHOULDER,
        item_level=120,
        phase=1,
        stats={"agility": 30, "stamina": 25, "crit_rating": 22, "hit_rating": 14},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"crit_rating": 3},
        set_id="netherblade",
    ),
    Item(
        id=31030,
        name="Slayer's Shoulderpads",
        slot=GearSlot.SHOULDER,
        item_level=146,
        phase=3,
        stats={"agility": 38, "stamina": 33, "crit_rating": 26, "hit_rating": 18},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"crit_rating": 3},
        set_id="slayer",
    ),
    # === Back ===
    Item(
        id=28672,
        name="Drape of the Dark Reavers",
        slot=GearSlot.BACK,
        item_level=115,
        phase=1,
        stats={"agility": 22, "stamina": 18, "crit_rating": 14, "hit_rating": 10},
    ),
    Item(
        id=32323,
        name="Shadowmoon Destroyer's Drape",
        slot=GearSlot.BACK,
        item_level=141,
        phase=3,
        stats={"agility": 28, "stamina": 22, "crit_rating": 22, "hit_rating": 14},
    ),
    # === Chest ===
    Item(
        id=29045,
        name="Netherblade Chestpiece",
        slot=GearSlot.CHEST,
        item_level=120,
        phase=1,
        stats={"agility": 42, "stamina": 34, "crit_rating": 30, "hit_rating": 20},
        sockets=[GemSlot.RED, GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 4},
        set_id="netherblade",
    ),
    Item(
        id=31028,
        name="Slayer's Chestguard",
        slot=GearSlot.CHEST,
        item_level=146,
        phase=3,
        stats={"agility": 50, "stamina": 42, "crit_rating": 36, "hit_rating": 24, "haste_rating": 16},
        sockets=[GemSlot.RED, GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 4},
        set_id="slayer",
    ),
    # === Wrist ===
    Item(
        id=29246,
        name="Nightfall Wristguards",
        slot=GearSlot.WRIST,
        item_level=115,
        phase=1,
        stats={"agility": 24, "stamina": 16, "crit_rating": 18},
    ),
    Item(
        id=32324,
        name="Insidious Bands",
        slot=GearSlot.WRIST,
        item_level=141,
        phase=3,
        stats={"agility": 30, "stamina": 22, "crit_rating": 22, "expertise_rating": 14},
    ),
    # === Hands ===
    Item(
        id=29047,
        name="Netherblade Gloves",
        slot=GearSlot.HANDS,
        item_level=120,
        phase=1,
        stats={"agility": 34, "stamina": 27, "crit_rating": 22, "hit_rating": 16},
        sockets=[GemSlot.RED],
        socket_bonus={"agility": 2},
        set_id="netherblade",
    ),
    Item(
        id=31026,
        name="Slayer's Handguards",
        slot=GearSlot.HANDS,
        item_level=146,
        phase=3,
        stats={"agility": 42, "stamina": 36, "crit_rating": 28, "hit_rating": 20, "expertise_rating": 14},
        sockets=[GemSlot.RED],
        socket_bonus={"agility": 2},
        set_id="slayer",
    ),
    # === Waist ===
    Item(
        id=28828,
        name="Gronn-Stitched Girdle",
        slot=GearSlot.WAIST,
        item_level=115,
        phase=1,
        stats={"agility": 28, "stamina": 22, "crit_rating": 20, "hit_rating": 14},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    Item(
        id=32348,
        name="Belt of One-Hundred Deaths",
        slot=GearSlot.WAIST,
        item_level=141,
        phase=3,
        stats={"agility": 36, "stamina": 28, "crit_rating": 26, "expertise_rating": 18},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    # === Legs ===
    Item(
        id=29046,
        name="Netherblade Breeches",
        slot=GearSlot.LEGS,
        item_level=120,
        phase=1,
        stats={"agility": 38, "stamina": 30, "crit_rating": 26, "hit_rating": 18},
        sockets=[GemSlot.RED, GemSlot.YELLOW, GemSlot.BLUE],
        socket_bonus={"agility": 4},
        set_id="netherblade",
    ),
    Item(
        id=31029,
        name="Slayer's Legguards",
        slot=GearSlot.LEGS,
        item_level=146,
        phase=3,
        stats={"agility": 46, "stamina": 39, "crit_rating": 32, "hit_rating": 22, "haste_rating": 14},
        sockets=[GemSlot.RED, GemSlot.YELLOW, GemSlot.BLUE],
        socket_bonus={"agility": 4},
        set_id="slayer",
    ),
    # === Feet ===
    Item(
        id=28545,
        name="Edgewalker Longboots",
        slot=GearSlot.FEET,
        item_level=115,
        phase=1,
        stats={"agility": 30, "stamina": 22, "crit_rating": 18, "hit_rating": 16},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    Item(
        id=32366,
        name="Shadowmaster's Boots",
        slot=GearSlot.FEET,
        item_level=141,
        phase=3,
        stats={"agility": 38, "stamina": 28, "crit_rating": 24, "hit_rating": 18, "haste_rating": 12},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    # === Rings ===
    Item(
        id=28757,
        name="Ring of a Thousand Marks",
        slot=GearSlot.RING_1,
        item_level=115,
        phase=1,
        stats={"agility": 22, "stamina": 18, "crit_rating": 14, "hit_rating": 10},
    ),
    Item(
        id=30834,
        name="Shapeshifter's Signet",
        slot=GearSlot.RING_1,
        item_level=128,
        phase=2,
        stats={"agility": 26, "stamina": 20, "crit_rating": 18, "hit_rating": 14},
    ),
    Item(
        id=28649,
        name="Garona's Signet Ring",
        slot=GearSlot.RING_2,
        item_level=115,
        phase=1,
        stats={"agility": 20, "stamina": 16, "crit_rating": 18, "expertise_rating": 10},
    ),
    Item(
        id=32497,
        name="Stormrage Signet Ring",
        slot=GearSlot.RING_2,
        item_level=141,
        phase=3,
        stats={"agility": 28, "stamina": 22, "crit_rating": 22, "hit_rating": 16},
    ),
    # === Ranged ===
    Item(
        id=28772,
        name="Sunfury Bow of the Phoenix",
        slot=GearSlot.RANGED,
        item_level=115,
        phase=1,
        stats={"agility": 15, "stamina": 10, "crit_rating": 12, "hit_rating": 8},
    ),
    Item(
        id=30724,
        name="Barrel-Blade Longrifle",
        slot=GearSlot.RANGED,
        item_level=128,
        phase=2,
        stats={"agility": 18, "stamina": 12, "crit_rating": 14, "hit_rating": 10},
    ),
    Item(
        id=34334,
        name="Thori'dal, the Stars' Fury",
        slot=GearSlot.RANGED,
        item_level=164,
        phase=5,
        stats={"agility": 24, "stamina": 18, "crit_rating": 18, "haste_rating": 14},
    ),
    # === Extra P4/P5 items to ensure phase spread ===
    Item(
        id=33484,
        name="Ebon Mask",
        slot=GearSlot.HEAD,
        item_level=141,
        phase=4,
        stats={"agility": 44, "stamina": 36, "crit_rating": 28, "hit_rating": 20},
        sockets=[GemSlot.META, GemSlot.RED],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=33496,
        name="Nether Shadow Tunic",
        slot=GearSlot.CHEST,
        item_level=141,
        phase=4,
        stats={"agility": 46, "stamina": 38, "crit_rating": 32, "hit_rating": 20, "expertise_rating": 14},
        sockets=[GemSlot.RED, GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=34448,
        name="Coif of Alleria",
        slot=GearSlot.HEAD,
        item_level=159,
        phase=5,
        stats={"agility": 54, "stamina": 46, "crit_rating": 34, "haste_rating": 26},
        sockets=[GemSlot.META, GemSlot.RED],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=33503,
        name="Waistguard of the Great Beast",
        slot=GearSlot.WAIST,
        item_level=141,
        phase=4,
        stats={"agility": 34, "stamina": 26, "crit_rating": 24, "hit_rating": 16},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    # ===================================================================
    # Expanded item database — additional P1-P5 Rogue gear (~55 items)
    # ===================================================================
    # --- Main Hand Weapons ---
    Item(
        id=28438,
        name="Dragonstrike",
        slot=GearSlot.MAIN_HAND,
        item_level=136,
        phase=1,
        stats={"agility": 20, "hit_rating": 14, "haste_rating": 16},
        weapon=WeaponStats(min_damage=137, max_damage=255, speed=2.6, weapon_type=WeaponType.FIST, dps=75.4),
        proc=ProcEffect(trigger=ProcTrigger.ON_HIT, rate=0.04, duration=10.0, effect={"haste_rating": 240.0}),
    ),
    Item(
        id=28297,
        name="Gladiator's Slicer",
        slot=GearSlot.MAIN_HAND,
        item_level=123,
        phase=1,
        stats={"agility": 18, "stamina": 30, "crit_rating": 16, "hit_rating": 10},
        weapon=WeaponStats(min_damage=121, max_damage=226, speed=2.6, weapon_type=WeaponType.SWORD, dps=66.7),
    ),
    Item(
        id=29348,
        name="Twinblade of the Phoenix",
        slot=GearSlot.MAIN_HAND,
        item_level=128,
        phase=2,
        stats={"agility": 20, "stamina": 22, "crit_rating": 20, "haste_rating": 14},
        weapon=WeaponStats(min_damage=138, max_damage=257, speed=2.7, weapon_type=WeaponType.SWORD, dps=73.1),
    ),
    Item(
        id=30311,
        name="Warp Slicer",
        slot=GearSlot.MAIN_HAND,
        item_level=128,
        phase=2,
        stats={"agility": 18, "stamina": 18, "hit_rating": 16, "expertise_rating": 16},
        weapon=WeaponStats(min_damage=132, max_damage=246, speed=2.6, weapon_type=WeaponType.SWORD, dps=72.7),
    ),
    Item(
        id=32946,
        name="Swiftsteel Bludgeon",
        slot=GearSlot.MAIN_HAND,
        item_level=141,
        phase=3,
        stats={"agility": 24, "stamina": 22, "crit_rating": 22, "haste_rating": 18},
        weapon=WeaponStats(min_damage=154, max_damage=287, speed=2.4, weapon_type=WeaponType.MACE, dps=91.9),
    ),
    Item(
        id=32262,
        name="Syphon of the Nathrezim",
        slot=GearSlot.MAIN_HAND,
        item_level=141,
        phase=3,
        stats={"agility": 22, "stamina": 24, "crit_rating": 20, "hit_rating": 16},
        weapon=WeaponStats(min_damage=91, max_damage=169, speed=1.8, weapon_type=WeaponType.DAGGER, dps=72.2),
    ),
    Item(
        id=34169,
        name="Fang of Kalecgos",
        slot=GearSlot.MAIN_HAND,
        item_level=159,
        phase=5,
        stats={"agility": 28, "stamina": 28, "crit_rating": 26, "haste_rating": 22},
        weapon=WeaponStats(min_damage=108, max_damage=201, speed=1.8, weapon_type=WeaponType.DAGGER, dps=85.8),
    ),
    Item(
        id=34331,
        name="Hand of the Deceiver",
        slot=GearSlot.MAIN_HAND,
        item_level=159,
        phase=5,
        stats={"agility": 26, "stamina": 26, "hit_rating": 20, "expertise_rating": 22},
        weapon=WeaponStats(min_damage=219, max_damage=408, speed=2.6, weapon_type=WeaponType.FIST, dps=120.6),
    ),
    # --- Off Hand Weapons ---
    Item(
        id=28768,
        name="Malchazeen",
        slot=GearSlot.OFF_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 16, "stamina": 14, "crit_rating": 18, "hit_rating": 12},
        weapon=WeaponStats(min_damage=66, max_damage=122, speed=1.5, weapon_type=WeaponType.DAGGER, dps=62.7),
    ),
    Item(
        id=28392,
        name="Spiteblade",
        slot=GearSlot.OFF_HAND,
        item_level=115,
        phase=1,
        stats={"agility": 14, "stamina": 16, "hit_rating": 14, "expertise_rating": 10},
        weapon=WeaponStats(min_damage=78, max_damage=146, speed=1.6, weapon_type=WeaponType.SWORD, dps=70.0),
    ),
    Item(
        id=28439,
        name="Dragonstrike OH",
        slot=GearSlot.OFF_HAND,
        item_level=136,
        phase=1,
        stats={"agility": 20, "hit_rating": 14, "haste_rating": 16},
        weapon=WeaponStats(min_damage=71, max_damage=133, speed=1.4, weapon_type=WeaponType.FIST, dps=72.9),
        proc=ProcEffect(trigger=ProcTrigger.ON_HIT, rate=0.04, duration=10.0, effect={"haste_rating": 240.0}),
    ),
    Item(
        id=29533,
        name="Fist of Reckoning",
        slot=GearSlot.OFF_HAND,
        item_level=128,
        phase=2,
        stats={"agility": 18, "stamina": 20, "crit_rating": 16, "hit_rating": 12},
        weapon=WeaponStats(min_damage=80, max_damage=150, speed=1.5, weapon_type=WeaponType.FIST, dps=76.7),
    ),
    Item(
        id=32369,
        name="Blade of Savagery",
        slot=GearSlot.OFF_HAND,
        item_level=141,
        phase=3,
        stats={"agility": 20, "stamina": 22, "crit_rating": 18, "haste_rating": 16},
        weapon=WeaponStats(min_damage=95, max_damage=177, speed=1.5, weapon_type=WeaponType.SWORD, dps=90.7),
    ),
    Item(
        id=34203,
        name="Shiv of Exsanguination",
        slot=GearSlot.OFF_HAND,
        item_level=154,
        phase=5,
        stats={"agility": 22, "stamina": 24, "crit_rating": 22, "expertise_rating": 18},
        weapon=WeaponStats(min_damage=130, max_damage=243, speed=1.5, weapon_type=WeaponType.DAGGER, dps=124.3),
    ),
    Item(
        id=34170,
        name="Quel'Serrar",
        slot=GearSlot.OFF_HAND,
        item_level=154,
        phase=5,
        stats={"agility": 20, "stamina": 26, "hit_rating": 22, "haste_rating": 20},
        weapon=WeaponStats(min_damage=134, max_damage=250, speed=1.5, weapon_type=WeaponType.SWORD, dps=128.0),
    ),
    # --- Trinkets ---
    Item(
        id=23206,
        name="Mark of the Champion",
        slot=GearSlot.TRINKET_1,
        item_level=92,
        phase=1,
        stats={"attack_power": 150},
    ),
    Item(
        id=28034,
        name="Hourglass of the Unraveller",
        slot=GearSlot.TRINKET_2,
        item_level=115,
        phase=1,
        stats={"crit_rating": 32},
        proc=ProcEffect(
            trigger=ProcTrigger.ON_CRIT,
            rate=0.10,
            icd=50.0,
            duration=10.0,
            effect={"attack_power": 300.0},
        ),
    ),
    Item(
        id=28288,
        name="Abacus of Violent Odds",
        slot=GearSlot.TRINKET_2,
        item_level=115,
        phase=1,
        on_use=ProcEffect(
            trigger=ProcTrigger.ON_USE,
            rate=1.0,
            icd=120.0,
            duration=10.0,
            effect={"haste_rating": 260.0},
        ),
    ),
    Item(
        id=28579,
        name="Romulo's Poison Vial",
        slot=GearSlot.TRINKET_2,
        item_level=115,
        phase=1,
        proc=ProcEffect(
            trigger=ProcTrigger.PPM,
            rate=1.0,
            icd=0.0,
            duration=0.0,
            effect={"nature_damage": 222.0},
        ),
    ),
    Item(
        id=33832,
        name="Stonebreaker's Totem",
        slot=GearSlot.TRINKET_2,
        item_level=141,
        phase=4,
        stats={"attack_power": 84},
        on_use=ProcEffect(
            trigger=ProcTrigger.ON_USE,
            rate=1.0,
            icd=120.0,
            duration=20.0,
            effect={"attack_power": 320.0},
        ),
    ),
    Item(
        id=34427,
        name="Blackened Naaru Sliver",
        slot=GearSlot.TRINKET_1,
        item_level=154,
        phase=5,
        proc=ProcEffect(
            trigger=ProcTrigger.ON_HIT,
            rate=0.10,
            icd=45.0,
            duration=20.0,
            effect={"attack_power": 44.0},
            stacks=10,
        ),
    ),
    # --- Head ---
    Item(
        id=28182,
        name="Helm of the Fallen Hero",
        slot=GearSlot.HEAD,
        item_level=115,
        phase=1,
        stats={"agility": 36, "stamina": 28, "crit_rating": 24, "hit_rating": 14},
        sockets=[GemSlot.META, GemSlot.RED],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=32235,
        name="Cursed Vision of Sargeras",
        slot=GearSlot.HEAD,
        item_level=151,
        phase=3,
        stats={"agility": 50, "stamina": 40, "hit_rating": 24, "haste_rating": 20},
        sockets=[GemSlot.META, GemSlot.RED],
        socket_bonus={"agility": 4},
    ),
    # --- Neck ---
    Item(
        id=29381,
        name="Choker of Vile Intent",
        slot=GearSlot.NECK,
        item_level=120,
        phase=1,
        stats={"agility": 24, "stamina": 18, "crit_rating": 18, "hit_rating": 12},
    ),
    Item(
        id=32260,
        name="Pendant of the Perilous",
        slot=GearSlot.NECK,
        item_level=141,
        phase=3,
        stats={"agility": 28, "stamina": 22, "crit_rating": 20, "hit_rating": 16},
    ),
    Item(
        id=34358,
        name="Collar of the Pit Lord",
        slot=GearSlot.NECK,
        item_level=154,
        phase=5,
        stats={"agility": 32, "stamina": 26, "crit_rating": 22, "expertise_rating": 20},
    ),
    # --- Shoulder ---
    Item(
        id=27797,
        name="Wastewalker Shoulderpads",
        slot=GearSlot.SHOULDER,
        item_level=108,
        phase=1,
        stats={"agility": 26, "stamina": 21, "crit_rating": 18, "hit_rating": 12},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    Item(
        id=34192,
        name="Pauldrons of Perseverance",
        slot=GearSlot.SHOULDER,
        item_level=154,
        phase=5,
        stats={"agility": 42, "stamina": 36, "crit_rating": 30, "expertise_rating": 22},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"crit_rating": 3},
    ),
    Item(
        id=30055,
        name="Shoulderpads of the Stranger",
        slot=GearSlot.SHOULDER,
        item_level=128,
        phase=2,
        stats={"agility": 32, "stamina": 28, "crit_rating": 22, "hit_rating": 16},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    # --- Back ---
    Item(
        id=28256,
        name="Cloak of the Pit Stalker",
        slot=GearSlot.BACK,
        item_level=112,
        phase=1,
        stats={"agility": 20, "stamina": 16, "crit_rating": 12, "hit_rating": 8},
    ),
    Item(
        id=30098,
        name="Razor-Scale Battlecloak",
        slot=GearSlot.BACK,
        item_level=128,
        phase=2,
        stats={"agility": 24, "stamina": 20, "crit_rating": 18, "hit_rating": 12},
    ),
    Item(
        id=34241,
        name="Cloak of Blade Turning",
        slot=GearSlot.BACK,
        item_level=154,
        phase=5,
        stats={"agility": 32, "stamina": 26, "crit_rating": 24, "haste_rating": 16},
    ),
    # --- Chest ---
    Item(
        id=28228,
        name="Chestguard of the Conniver",
        slot=GearSlot.CHEST,
        item_level=115,
        phase=1,
        stats={"agility": 38, "stamina": 30, "crit_rating": 26, "hit_rating": 16},
        sockets=[GemSlot.RED, GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=30222,
        name="Nether Shadow Tunic P2",
        slot=GearSlot.CHEST,
        item_level=128,
        phase=2,
        stats={"agility": 44, "stamina": 34, "crit_rating": 30, "hit_rating": 18},
        sockets=[GemSlot.RED, GemSlot.YELLOW, GemSlot.BLUE],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=34397,
        name="Bladed Chaos Tunic",
        slot=GearSlot.CHEST,
        item_level=159,
        phase=5,
        stats={"agility": 56, "stamina": 46, "crit_rating": 38, "haste_rating": 24, "expertise_rating": 18},
        sockets=[GemSlot.RED, GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 4},
    ),
    # --- Wrist ---
    Item(
        id=28174,
        name="Skulker's Greaves Bracers",
        slot=GearSlot.WRIST,
        item_level=112,
        phase=1,
        stats={"agility": 22, "stamina": 14, "crit_rating": 14},
    ),
    Item(
        id=30041,
        name="Vambraces of Ending",
        slot=GearSlot.WRIST,
        item_level=128,
        phase=2,
        stats={"agility": 28, "stamina": 18, "crit_rating": 20, "hit_rating": 12},
    ),
    Item(
        id=34447,
        name="Bracers of the Forgotten Conqueror",
        slot=GearSlot.WRIST,
        item_level=154,
        phase=5,
        stats={"agility": 34, "stamina": 24, "crit_rating": 24, "haste_rating": 18},
    ),
    # --- Hands ---
    Item(
        id=27509,
        name="Handgrips of Assassination",
        slot=GearSlot.HANDS,
        item_level=108,
        phase=1,
        stats={"agility": 30, "stamina": 24, "crit_rating": 18, "hit_rating": 12},
        sockets=[GemSlot.RED],
        socket_bonus={"agility": 2},
    ),
    Item(
        id=34370,
        name="Gloves of Immortal Dusk",
        slot=GearSlot.HANDS,
        item_level=154,
        phase=5,
        stats={"agility": 46, "stamina": 38, "crit_rating": 32, "expertise_rating": 20, "haste_rating": 16},
        sockets=[GemSlot.RED],
        socket_bonus={"agility": 2},
    ),
    # --- Waist ---
    Item(
        id=28750,
        name="Girdle of Treachery",
        slot=GearSlot.WAIST,
        item_level=115,
        phase=1,
        stats={"agility": 26, "stamina": 20, "crit_rating": 18, "hit_rating": 12},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    Item(
        id=34558,
        name="Cord of Screaming Terrors",
        slot=GearSlot.WAIST,
        item_level=154,
        phase=5,
        stats={"agility": 38, "stamina": 30, "crit_rating": 28, "haste_rating": 20},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    # --- Legs ---
    Item(
        id=28741,
        name="Skulker's Greaves",
        slot=GearSlot.LEGS,
        item_level=115,
        phase=1,
        stats={"agility": 34, "stamina": 26, "crit_rating": 24, "hit_rating": 16},
        sockets=[GemSlot.RED, GemSlot.YELLOW, GemSlot.BLUE],
        socket_bonus={"agility": 4},
    ),
    Item(
        id=34188,
        name="Leggings of Immortal Night",
        slot=GearSlot.LEGS,
        item_level=154,
        phase=5,
        stats={"agility": 50, "stamina": 42, "crit_rating": 36, "haste_rating": 22, "expertise_rating": 16},
        sockets=[GemSlot.RED, GemSlot.YELLOW, GemSlot.BLUE],
        socket_bonus={"agility": 4},
    ),
    # --- Feet ---
    Item(
        id=28517,
        name="Boots of Elusion",
        slot=GearSlot.FEET,
        item_level=115,
        phase=1,
        stats={"agility": 28, "stamina": 20, "crit_rating": 16, "hit_rating": 14},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    Item(
        id=30104,
        name="Cobra-Lash Boots",
        slot=GearSlot.FEET,
        item_level=128,
        phase=2,
        stats={"agility": 34, "stamina": 24, "crit_rating": 22, "hit_rating": 16},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    Item(
        id=34575,
        name="Boots of the Forgotten Protector",
        slot=GearSlot.FEET,
        item_level=159,
        phase=5,
        stats={"agility": 42, "stamina": 32, "crit_rating": 28, "haste_rating": 20},
        sockets=[GemSlot.RED, GemSlot.YELLOW],
        socket_bonus={"agility": 3},
    ),
    # --- Rings ---
    Item(
        id=28247,
        name="Ethereal Band of the Sentry",
        slot=GearSlot.RING_1,
        item_level=112,
        phase=1,
        stats={"agility": 20, "stamina": 15, "crit_rating": 12, "hit_rating": 8},
    ),
    Item(
        id=29301,
        name="Band of the Eternal Champion",
        slot=GearSlot.RING_1,
        item_level=120,
        phase=1,
        stats={"agility": 24, "stamina": 19, "crit_rating": 16, "hit_rating": 12},
    ),
    Item(
        id=30052,
        name="Ring of Lethality",
        slot=GearSlot.RING_2,
        item_level=128,
        phase=2,
        stats={"agility": 24, "stamina": 18, "crit_rating": 18, "expertise_rating": 12},
    ),
    Item(
        id=34361,
        name="Band of Ruinous Delight",
        slot=GearSlot.RING_2,
        item_level=154,
        phase=5,
        stats={"agility": 30, "stamina": 24, "crit_rating": 22, "haste_rating": 18},
    ),
    Item(
        id=34189,
        name="Band of the Eternal Defender",
        slot=GearSlot.RING_1,
        item_level=154,
        phase=5,
        stats={"agility": 32, "stamina": 26, "crit_rating": 24, "hit_rating": 18},
    ),
    # --- Ranged ---
    Item(
        id=28504,
        name="Steelhawk Crossbow",
        slot=GearSlot.RANGED,
        item_level=115,
        phase=1,
        stats={"agility": 14, "stamina": 10, "crit_rating": 10, "hit_rating": 8},
    ),
    Item(
        id=32336,
        name="Black Bow of the Betrayer",
        slot=GearSlot.RANGED,
        item_level=151,
        phase=3,
        stats={"agility": 22, "stamina": 16, "crit_rating": 16, "hit_rating": 14},
    ),
]

# ---------------------------------------------------------------------------
# Curated enchants
# ---------------------------------------------------------------------------

_CURATED_ENCHANTS: list[Enchant] = [
    Enchant(
        id=2673,
        name="Mongoose",
        slot=GearSlot.MAIN_HAND,
        proc=ProcEffect(
            trigger=ProcTrigger.PPM,
            rate=1.0,
            duration=15.0,
            effect={"agility": 120.0, "haste_pct": 2.0},
        ),
    ),
    Enchant(
        id=2674,
        name="Mongoose",
        slot=GearSlot.OFF_HAND,
        proc=ProcEffect(
            trigger=ProcTrigger.PPM,
            rate=1.0,
            duration=15.0,
            effect={"agility": 120.0, "haste_pct": 2.0},
        ),
    ),
    Enchant(
        id=3225,
        name="Executioner",
        slot=GearSlot.MAIN_HAND,
        proc=ProcEffect(
            trigger=ProcTrigger.PPM,
            rate=1.0,
            duration=15.0,
            effect={"armor_penetration": 840.0},
        ),
    ),
    Enchant(
        id=2670,
        name="Greater Agility",
        slot=GearSlot.BACK,
        stats={"agility": 12},
    ),
    Enchant(
        id=2564,
        name="Enchant Boots - Cat's Swiftness",
        slot=GearSlot.FEET,
        stats={"agility": 6},
    ),
    Enchant(
        id=2617,
        name="Enchant Bracer - Assault",
        slot=GearSlot.WRIST,
        stats={"attack_power": 24},
    ),
    Enchant(
        id=3860,
        name="Enchant Gloves - Superior Agility",
        slot=GearSlot.HANDS,
        stats={"agility": 15},
    ),
    Enchant(
        id=3012,
        name="Enchant Chest - Exceptional Stats",
        slot=GearSlot.CHEST,
        stats={"agility": 6, "stamina": 6, "strength": 6},
    ),
    Enchant(
        id=3010,
        name="Enchant Head - Glyph of Ferocity",
        slot=GearSlot.HEAD,
        stats={"agility": 34, "hit_rating": 16},
    ),
    Enchant(
        id=2996,
        name="Enchant Shoulder - Greater Inscription of Vengeance",
        slot=GearSlot.SHOULDER,
        stats={"attack_power": 30, "crit_rating": 10},
    ),
    Enchant(
        id=3013,
        name="Enchant Legs - Nethercobra Leg Armor",
        slot=GearSlot.LEGS,
        stats={"attack_power": 50, "crit_rating": 12},
    ),
]

# ---------------------------------------------------------------------------
# Curated gems
# ---------------------------------------------------------------------------

_CURATED_GEMS: list[Gem] = [
    Gem(id=24028, name="Delicate Living Ruby", color=GemSlot.RED, stats={"agility": 8}),
    Gem(id=24055, name="Shifting Nightseye", color=GemSlot.BLUE, stats={"agility": 4, "stamina": 6}),
    Gem(
        id=32409,
        name="Relentless Earthstorm Diamond",
        color=GemSlot.META,
        stats={"agility": 12, "crit_damage_pct": 3},
        meta_condition="More red than blue",
    ),
    Gem(id=24048, name="Smooth Dawnstone", color=GemSlot.YELLOW, stats={"crit_rating": 8}),
    Gem(id=24058, name="Glinting Noble Topaz", color=GemSlot.YELLOW, stats={"agility": 4, "hit_rating": 4}),
    Gem(id=24054, name="Wicked Noble Topaz", color=GemSlot.YELLOW, stats={"agility": 4, "crit_rating": 4}),
]

# ---------------------------------------------------------------------------
# Curated set bonuses
# ---------------------------------------------------------------------------

_CURATED_SET_BONUSES: list[SetBonus] = [
    SetBonus(set_name="netherblade", pieces_required=2, effect={"snd_bonus_seconds": 3}),
    SetBonus(set_name="netherblade", pieces_required=4, effect={"finisher_cp_proc_chance": 0.15}),
    SetBonus(set_name="slayer", pieces_required=2, effect={"snd_haste_bonus_pct": 0.05}),
    SetBonus(set_name="slayer", pieces_required=4, effect={"builder_damage_bonus_pct": 0.06}),
]


# ---------------------------------------------------------------------------
# ItemDatabase
# ---------------------------------------------------------------------------


class ItemDatabase:
    """In-memory item database loaded from hardcoded curated data."""

    def __init__(self) -> None:
        self._items: dict[int, Item] = {}
        self._enchants: dict[int, Enchant] = {}
        self._gems: dict[int, Gem] = {}
        self._set_bonuses: dict[str, list[SetBonus]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load curated item data into memory."""
        for item in _CURATED_ITEMS:
            self._items[item.id] = item
        for enchant in _CURATED_ENCHANTS:
            self._enchants[enchant.id] = enchant
        for gem in _CURATED_GEMS:
            self._gems[gem.id] = gem
        for bonus in _CURATED_SET_BONUSES:
            self._set_bonuses.setdefault(bonus.set_name, []).append(bonus)

    def get_item(self, item_id: int) -> Item | None:
        """Return an item by its ID, or None if not found."""
        return self._items.get(item_id)

    def get_enchant(self, enchant_id: int) -> Enchant | None:
        """Return an enchant by its ID, or None if not found."""
        return self._enchants.get(enchant_id)

    def get_gem(self, gem_id: int) -> Gem | None:
        """Return a gem by its ID, or None if not found."""
        return self._gems.get(gem_id)

    def get_set_bonuses(self, set_name: str) -> list[SetBonus]:
        """Return all set bonuses for a set, or an empty list."""
        return self._set_bonuses.get(set_name, [])

    def items_for_slot(self, slot: GearSlot, *, phase: int | None = None, min_ilvl: int = 0) -> list[Item]:
        """Return items for a slot, optionally filtered by phase and min item level."""
        results = [i for i in self._items.values() if i.slot == slot and i.item_level >= min_ilvl]
        if phase is not None:
            results = [i for i in results if i.phase <= phase]
        return sorted(results, key=lambda x: x.item_level, reverse=True)

    def search(self, query: str, *, slot: GearSlot | None = None) -> list[Item]:
        """Search items by name substring (case-insensitive), optionally filtered by slot."""
        q = query.lower()
        results = [i for i in self._items.values() if q in i.name.lower()]
        if slot is not None:
            results = [i for i in results if i.slot == slot]
        return results

    def rogue_items_for_slot(self, slot: GearSlot, spec: RogueSpec, phase: int = 5) -> list[Item]:
        """Return items suitable for a Rogue spec in a slot up to a phase."""
        return self.items_for_slot(slot, phase=phase)
