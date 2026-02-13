# Phase 4: Validation — WoWSims WASM Oracle + WCL Log Comparison

## Overview

Triple validation strategy: (1) automated 2% regression against WoWSims WASM, (2) Warcraft Logs parse comparison for realism checks, (3) unit tests for individual mechanics.

## Module: `sim/validation.py`

### 1. WoWSims WASM Oracle

**Setup:**
- Clone `wowsims/tbc` repo, build Go→WASM binary via their Makefile
- Store binary in `sim/data/wowsims.wasm`
- Use `wasmtime-py` (Python bindings for Wasmtime WASM runtime) to invoke

**Config translation:**
- Our `SimConfig` → WoWSims `RaidSimRequest` protobuf
- Item IDs are shared (both use WoW item IDs)
- Talent strings are identical format ("20/41/0")
- Buff mapping: our buff IDs → WoWSims protobuf enum values (manual mapping table)
- Boss config: armor + debuffs map directly

**Invocation flow:**
1. Serialize `RaidSimRequest` protobuf to bytes
2. Call WASM `runRaidSim(input_bytes)` via wasmtime
3. Deserialize `RaidSimResult` protobuf from output bytes
4. Extract `raidMetrics.dps.avg` as reference DPS

### 2. Regression Test Suite

**Profiles (10 canonical setups):**

| # | Profile | Spec | Phase | Key Gear |
|---|---------|------|-------|----------|
| 1 | P1 BiS Combat Swords | combat_swords | 1 | Blinkstrike + Latro's, DST + Brooch |
| 2 | P2 BiS Combat Swords | combat_swords | 2 | Netherblade 4pc, MH Talon of Azshara |
| 3 | P3 BiS Combat Swords | combat_swords | 3 | Deathmantle 2pc, Warglaives |
| 4 | P5 BiS Combat Swords | combat_swords | 5 | Slayer's 4pc, Warglaives, Madness |
| 5 | P1 BiS Combat Daggers | combat_daggers | 1 | Emerald Ripper + Latro's |
| 6 | P3 BiS Combat Fists | combat_fists | 3 | Fist of the Deity, etc. |
| 7 | P3 BiS Mutilate | assassination_mutilate | 3 | Dual daggers, T5 mix |
| 8 | P5 BiS Mutilate | assassination_mutilate | 5 | Slayer's, BT/Hyjal daggers |
| 9 | No WF group (Combat) | combat_swords | 3 | Same as #3, no WF totem |
| 10 | Solo (no buffs) | combat_swords | 3 | Same as #3, no buffs/debuffs |

**Test structure:**
```python
@pytest.mark.parametrize("profile", VALIDATION_PROFILES)
async def test_wowsims_parity(profile, sim_runner, wowsims_oracle):
    our_result = await sim_runner.sim_run(profile.config)
    wowsims_dps = await wowsims_oracle.run(profile.config)
    drift_pct = abs(our_result.dps_mean - wowsims_dps) / wowsims_dps * 100
    assert drift_pct <= profile.tolerance_pct, (
        f"{profile.name}: {our_result.dps_mean:.1f} vs {wowsims_dps:.1f} "
        f"({drift_pct:.1f}% drift, tolerance {profile.tolerance_pct}%)"
    )
```

**Known exceptions:**
- Mutilate profiles may exceed 2% due to Envenom DP stack consumption difference
- Thistle Tea energy difference (100 vs 40) may cause ~1% divergence in long fights
- Tolerance for Mutilate profiles: 4% instead of 2%

### 3. Warcraft Logs Comparison

Uses existing `scraping/` WCL API integration.

**Patchwerk-style encounters for comparison:**
- Brutallus (Sunwell) — closest to pure stand-and-deliver
- Patchwerk (Naxxramas, if available in TBC era)
- Gruul (if filtered to post-shatter phase)

**Methodology:**
1. Fetch all Rogue parses for encounter, filter by spec and ilvl range
2. Take 25th-75th percentile (IQR) to exclude extreme outliers
3. Sim with matching ilvl gear template and full raid buffs
4. Check if sim DPS falls within IQR — not a hard gate, just a realism flag

**Report output:**
```
Brutallus (P5 Combat Swords):
  Sim DPS:     2,347
  WCL Median:  2,198
  WCL IQR:     1,987 - 2,412
  Status:      WITHIN IQR ✓
  Note:        Sim is 6.8% above median (expected — sim assumes perfect play)
```

### 4. Unit Tests for Individual Mechanics

**Separate from integration validation — these test formulas in isolation:**

```python
# test_mechanics.py
def test_white_hit_table_crit_pushoff():
    """Crit pushed off when miss+dodge+glance >= available space."""
    ...

def test_yellow_hit_table_no_pushoff():
    """Yellow crit on separate roll, never pushed off."""
    ...

def test_armor_reduction_standard_boss():
    """7700 armor boss = 42.2% DR with no debuffs."""
    ...

def test_armor_cap():
    """Armor can never reduce more than 75%."""
    ...

def test_glancing_multiplier():
    """Flat 0.75x vs level 73 boss."""
    ...

def test_energy_tick_amount():
    """20.2 energy per tick, not 20.0."""
    ...

def test_sword_spec_icd():
    """500ms ICD prevents chain procs."""
    ...
```

### 5. CI Integration

```yaml
# In CI pipeline
- name: Run sim validation
  run: |
    python3 -m pytest tests/unit/sim/ -v
    python3 -m pytest tests/integration/sim/ -v -m validation
```

Unit tests run always. Validation tests (requiring WASM binary) run in integration suite.

## Dependencies

- `wasmtime` — Python WASM runtime (pip install wasmtime)
- `protobuf` — for WoWSims proto format (pip install protobuf)
- WoWSims WASM binary built from source
- WCL API access (existing integration)
