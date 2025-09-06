# 🚨 CRITICAL DATA WARNING - READ FIRST! 🚨

## ⚠️ **MANDATORY FOR ALL AGENTS - NO EXCEPTIONS!**

### **THE TOY DATA TRAP - AVOID AT ALL COSTS!**

**❌ NEVER USE THESE FILES FOR REAL ANALYSIS:**
- `data/ipta_dr2/processed/cosmic_string_inputs_versionA.npz` - **TOY DATA!**
- `data/ipta_dr2/processed/cosmic_string_inputs_versionB.npz` - **TOY DATA!**
- Any file with "cosmic_string_inputs" in the name - **SIMULATION DATA!**

**✅ ONLY USE REAL DATA:**
- `data/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA/` - **REAL IPTA DR2 DATA!**
- `data/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionB/` - **REAL IPTA DR2 DATA!**

---

## 🎯 **WHAT HAPPENED (AGENT DRIFT PREVENTION):**

1. **Agent used `cosmic_string_inputs_versionA.npz`** - This is SIMULATED data for testing cosmic string detection
2. **Forensic system correctly flagged it as `TOY_DATA`** - The system worked perfectly!
3. **Agent misinterpreted the results** - Thought the data was real when it was simulation
4. **31.7% clustering found** - This is in SIMULATED data, not real observations

---

## 🔒 **MANDATORY PROTOCOLS:**

### **1. DATA VERIFICATION CHECKLIST:**
- [ ] Check file name - does it contain "cosmic_string_inputs"? → **STOP! TOY DATA!**
- [ ] Check source directory - is it in `real_ipta_dr2/`? → **GOOD! REAL DATA!**
- [ ] Run forensic disproof - if `TOY_DATA` verdict → **STOP! USE REAL DATA!**
- [ ] Verify pulsar names - should be J+coordinates from real IPTA DR2

### **2. REAL DATA PROCESSING:**
```bash
# CORRECT: Process real IPTA DR2 data
python process_real_ipta_data.py  # Uses real_ipta_dr2/ directory

# WRONG: Using cosmic string inputs
# python REAL_ENHANCED_COSMIC_STRING_SYSTEM.py  # Uses cosmic_string_inputs (TOY!)
```

### **3. FORENSIC SYSTEM USAGE:**
- **ALWAYS run forensic disproof first**
- **If verdict = `TOY_DATA` → STOP and use real data**
- **If verdict = `STRONG` or `WEAK` → Continue analysis**

---

## 📁 **FILE STRUCTURE CLARIFICATION:**

```
data/
├── ipta_dr2/processed/
│   ├── cosmic_string_inputs_versionA.npz  ❌ TOY DATA - DO NOT USE!
│   ├── cosmic_string_inputs_versionB.npz  ❌ TOY DATA - DO NOT USE!
│   └── ipta_dr2_versionA_processed.npz   ⚠️  CHECK SOURCE!
└── real_ipta_dr2/ipta_par_files/DR2-master/release/
    ├── VersionA/  ✅ REAL IPTA DR2 DATA - USE THIS!
    └── VersionB/  ✅ REAL IPTA DR2 DATA - USE THIS!
```

---

## 🚀 **NEXT STEPS FOR REAL ANALYSIS:**

1. **Process real IPTA DR2 data** from `real_ipta_dr2/` directory
2. **Run forensic disproof** on real data results
3. **Look for real patterns** in real observations
4. **Document findings** with proper data source attribution

---

## ⚠️ **AGENT DRIFT PREVENTION:**

- **Read this file FIRST** before any data analysis
- **Verify data source** before running any system
- **Run forensic disproof** on ALL data before analysis
- **Document data source** in all results files
- **If in doubt, STOP and ask for clarification**

---

**REMEMBER: The forensic system is working perfectly - it caught the toy data! We just need to use the RIGHT data!** 🎯
