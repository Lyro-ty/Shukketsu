-- WCL API data tables (schema v5)
-- Tracked characters for personal monitoring
CREATE TABLE IF NOT EXISTS wcl_tracked_characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wcl_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    server_slug TEXT NOT NULL,
    server_region TEXT NOT NULL,
    endpoint TEXT NOT NULL DEFAULT 'fresh',
    class_id INTEGER,
    last_sync_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Encounter rankings (top 100 per encounter)
CREATE TABLE IF NOT EXISTS wcl_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    spec TEXT NOT NULL,
    dps REAL NOT NULL,
    duration_ms INTEGER NOT NULL,
    report_code TEXT NOT NULL,
    fight_id INTEGER NOT NULL,
    guild_name TEXT,
    server_name TEXT NOT NULL,
    server_region TEXT NOT NULL,
    faction INTEGER,
    raid_size INTEGER,
    bracket_data INTEGER,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(encounter_id, player_name, server_name, report_code)
);

-- Report metadata
CREATE TABLE IF NOT EXISTS wcl_reports (
    code TEXT PRIMARY KEY,
    title TEXT,
    zone_id INTEGER,
    zone_name TEXT,
    start_time INTEGER,
    end_time INTEGER,
    is_archived INTEGER NOT NULL DEFAULT 0,
    endpoint TEXT NOT NULL DEFAULT 'fresh',
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Fight metadata within reports
CREATE TABLE IF NOT EXISTS wcl_fights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT NOT NULL,
    kill INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    boss_percentage REAL,
    avg_item_level REAL,
    raid_size INTEGER,
    difficulty INTEGER,
    UNIQUE(report_code, fight_id)
);

-- CombatantInfo per player per fight
CREATE TABLE IF NOT EXISTS wcl_combatants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    spec_id INTEGER,
    faction INTEGER,
    strength INTEGER, agility INTEGER, stamina INTEGER,
    intellect INTEGER, spirit INTEGER,
    crit_melee INTEGER, crit_ranged INTEGER, crit_spell INTEGER,
    haste_melee INTEGER, haste_ranged INTEGER, haste_spell INTEGER,
    hit_melee INTEGER, hit_ranged INTEGER, hit_spell INTEGER,
    expertise INTEGER, dodge INTEGER, parry INTEGER, block INTEGER, armor INTEGER,
    gear_json TEXT,
    talents_json TEXT,
    auras_json TEXT,
    UNIQUE(report_code, fight_id, source_id)
);

-- Damage breakdown per player per fight
CREATE TABLE IF NOT EXISTS wcl_damage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    player_type TEXT,
    total_damage INTEGER NOT NULL,
    active_time_ms INTEGER,
    abilities_json TEXT,
    targets_json TEXT,
    UNIQUE(report_code, fight_id, player_name)
);

-- Buff uptimes per player per fight
CREATE TABLE IF NOT EXISTS wcl_buffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    buff_name TEXT NOT NULL,
    buff_guid INTEGER NOT NULL,
    total_uptime_ms INTEGER NOT NULL,
    total_uses INTEGER NOT NULL,
    bands_json TEXT,
    UNIQUE(report_code, fight_id, buff_guid)
);

-- Cast counts per player per fight
CREATE TABLE IF NOT EXISTS wcl_casts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    ability_name TEXT NOT NULL,
    cast_count INTEGER NOT NULL,
    UNIQUE(report_code, fight_id, player_name, ability_name)
);

-- Per-fight rankings (percentiles)
CREATE TABLE IF NOT EXISTS wcl_fight_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    server_name TEXT,
    class TEXT NOT NULL,
    spec TEXT,
    dps REAL NOT NULL,
    rank_percent INTEGER,
    total_parses INTEGER,
    UNIQUE(report_code, fight_id, player_name)
);

-- Character sync log
CREATE TABLE IF NOT EXISTS wcl_character_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_wcl_id INTEGER NOT NULL,
    report_code TEXT NOT NULL,
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT,
    dps REAL,
    rank_percent INTEGER,
    kill INTEGER,
    timestamp INTEGER,
    UNIQUE(character_wcl_id, report_code, fight_id)
);

CREATE INDEX IF NOT EXISTS idx_wcl_rankings_encounter ON wcl_rankings(encounter_id);
CREATE INDEX IF NOT EXISTS idx_wcl_rankings_player ON wcl_rankings(player_name, server_name);
CREATE INDEX IF NOT EXISTS idx_wcl_combatants_report ON wcl_combatants(report_code, fight_id);
CREATE INDEX IF NOT EXISTS idx_wcl_damage_report ON wcl_damage(report_code, fight_id);
CREATE INDEX IF NOT EXISTS idx_wcl_character_log_char ON wcl_character_log(character_wcl_id);
