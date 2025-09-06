# TOMORROW'S SPRINT PLAN (2h max)
## Cosmic String Detection Sensitivity Curve

### 🎯 **OBJECTIVES**
1. **Sweep Gμ = 5e-12 → 2e-11 in 5 steps** → plot detection fraction vs. Gμ
2. **Add --seed argument** → repeatable Monte-Carlo (100 skies per point → error bars)
3. **Push v0.3.0** with PNG curve + CSV data → publish-ready figure

### 🚀 **EXECUTION PLAN**

#### **Phase 1: Enhanced Injection Engine (30 min)**
```bash
# Test enhanced injection with seed
python enhanced_inject_cosmic_string_skies.py REAL_ENHANCED_COSMIC_STRING_RESULTS.json --Gmu 1e-11 --seed 42

# Verify reproducibility
python enhanced_inject_cosmic_string_skies.py REAL_ENHANCED_COSMIC_STRING_RESULTS.json --Gmu 1e-11 --seed 42
```

#### **Phase 2: Sensitivity Curve Generation (60 min)**
```bash
# Run full sensitivity curve
python generate_sensitivity_curve.py

# Verify results
python -c "import json; data=json.load(open('SENSITIVITY_CURVE_v0.3.0.json')); print('Gμ values:', [r['Gmu'] for r in data]); print('Detection fractions:', [r['mean_detection'] for r in data])"
```

#### **Phase 3: Git Commit & Tag (15 min)**
```bash
git add .
git commit -m "feat: sensitivity curve v0.3.0 — Monte-Carlo analysis with error bars"
git tag v0.3.0
```

#### **Phase 4: Verification (15 min)**
- Check PNG/PDF files generated
- Verify CSV data format
- Confirm error bars look reasonable
- Test reproducibility with same seeds

### 📊 **EXPECTED OUTPUTS**
- `SENSITIVITY_CURVE_v0.3.0.png` - Publication-ready figure
- `SENSITIVITY_CURVE_v0.3.0.pdf` - Vector version
- `SENSITIVITY_CURVE_v0.3.0.csv` - Data for analysis
- `SENSITIVITY_CURVE_v0.3.0.json` - Full results
- Git tag `v0.3.0` - Citable release

### 🎯 **SUCCESS CRITERIA**
- ✅ 5 Gμ values tested (5e-12 to 2e-11)
- ✅ 100 Monte-Carlo realizations per point
- ✅ Error bars on sensitivity curve
- ✅ Reproducible with same seeds
- ✅ Publication-ready figures
- ✅ Git tag v0.3.0 created

### 🚀 **NEXT STEPS AFTER SPRINT**
- Point the whole thing at real IPTA DR2
- Watch the sky confess
- Prepare for publication

---
**Ready to hunt cosmic strings!** 🌌🚀
