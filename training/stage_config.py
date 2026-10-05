"""
stage_config.py
================
Maps each of the 67 process features to its semiconductor
fabrication stage, and defines placeholder domain metadata (whether
a defect traced to that stage is typically reworkable, recyclable at
the material level, or scrap-only) used by the recommendation engine
in explain_model.py.

IMPORTANT - PLACEHOLDER DOMAIN KNOWLEDGE:
The `reworkable` flags, `disposition`, and `precaution` text below are
REASONABLE ENGINEERING PLACEHOLDERS based on general semiconductor
fab practice, not verified data from your specific fab line. Edit the
STAGE_METADATA dict below with your own process engineering knowledge
before treating recommendations as authoritative.

Author: ML Engineering Team
"""

# ------------------------------------------------------------------
# FEATURE -> STAGE MAPPING
# ------------------------------------------------------------------
# Maps every input feature to one of the 13 fabrication stages.
FEATURE_STAGE_MAP = {
    # Stage 1: Crystal Growth (Czochralski ingot growth)
    "Furnace_Temperature_Growth_C": "Crystal Growth",
    "Pull_Rate_mm_min": "Crystal Growth",
    "Crystal_Rotation_Speed_rpm": "Crystal Growth",
    "Oxygen_Concentration_ppma": "Crystal Growth",
    "Crystal_Diameter_mm": "Crystal Growth",
    "Resistivity_ohm_cm": "Crystal Growth",
    "Crystal_Defect_Count": "Crystal Growth",

    # Stage 2: Wafer Slicing
    "Diameter_Variation_um": "Wafer Slicing",
    "Surface_Crack_Score": "Wafer Slicing",
    "Blade_Speed_rpm": "Wafer Slicing",
    "Feed_Rate_mm_min": "Wafer Slicing",
    "Slice_Thickness_um": "Wafer Slicing",
    "Thickness_Variation_um": "Wafer Slicing",
    "Vibration_g": "Wafer Slicing",

    # Stage 3: Initial Polishing
    "Pad_Speed_Polish_rpm": "Initial Polishing",
    "Pressure_Polish_psi": "Initial Polishing",
    "Slurry_Flow_Polish_ml_min": "Initial Polishing",
    "Surface_Roughness_nm": "Initial Polishing",
    "Flatness_Polish_um": "Initial Polishing",

    # Stage 4: Cleaning
    "Water_Flow_lpm": "Cleaning",
    "Chemical_Concentration_pct": "Cleaning",
    "Cleaning_Temperature_C": "Cleaning",
    "Cleaning_Time_sec": "Cleaning",
    "Particle_Count_Clean": "Cleaning",

    # Stage 5: Oxidation
    "Furnace_Temperature_Oxidation_C": "Oxidation",
    "Oxygen_Flow_sccm": "Oxidation",
    "Oxidation_Time_min": "Oxidation",
    "Oxide_Thickness_nm": "Oxidation",
    "Uniformity_Oxidation_pct": "Oxidation",

    # Stage 6: Photoresist Coating
    "Spin_Speed_rpm": "Photoresist Coating",
    "Spin_Time_sec": "Photoresist Coating",
    "Resist_Thickness_um": "Photoresist Coating",
    "Soft_Bake_Temperature_C": "Photoresist Coating",

    # Stage 7: Photolithography (Exposure)
    "Exposure_Energy_mJ_cm2": "Photolithography",
    "Focus_Offset_um": "Photolithography",
    "Alignment_Error_um": "Photolithography",
    "Critical_Dimension_nm": "Photolithography",

    # Stage 8: Etching
    "RF_Power_W": "Etching",
    "Chamber_Pressure_Etch_mTorr": "Etching",
    "Gas_Flow_SF6_sccm": "Etching",
    "Gas_Flow_CF4_sccm": "Etching",
    "Gas_Flow_O2_sccm": "Etching",
    "Etch_Rate_nm_min": "Etching",

    # Stage 9: Ion Implantation
    "Beam_Current_mA": "Ion Implantation",
    "Implant_Dose_e15_ions_cm2": "Ion Implantation",
    "Implant_Energy_keV": "Ion Implantation",
    "Wafer_Temperature_Implant_C": "Ion Implantation",

    # Stage 10: Thin Film Deposition
    "Chamber_Temperature_Deposition_C": "Thin Film Deposition",
    "Chamber_Pressure_Deposition_mTorr": "Thin Film Deposition",
    "Gas_Flow_Deposition_sccm": "Thin Film Deposition",
    "Deposition_Rate_nm_min": "Thin Film Deposition",
    "Film_Thickness_nm": "Thin Film Deposition",

    # Stage 11: Chemical Mechanical Planarization (Final CMP)
    "CMP_Pad_Speed_rpm": "Final CMP",
    "CMP_Down_Force_psi": "Final CMP",
    "CMP_Slurry_Flow_ml_min": "Final CMP",
    "CMP_Removal_Rate_nm_min": "Final CMP",
    "CMP_Wafer_Flatness_nm": "Final CMP",

    # Stage 12: Final Metrology / Inspection
    "Film_Thickness_Error_nm": "Final Metrology",
    "Critical_Dimension_Error_nm": "Final Metrology",
    "Particle_Count_Final": "Final Metrology",
    "Defect_Count_Final": "Final Metrology",
    "Surface_Roughness_Final_nm": "Final Metrology",

    # Stage 13: Aggregate Quality Indices (derived / summary features,
    # not a physical process step - flagged separately below)
    "Defect_Probability": "Aggregate Quality Index",
    "Quality_Score": "Aggregate Quality Index",
    "Total_Particle_Count": "Aggregate Quality Index",
    "Process_Error_Index": "Aggregate Quality Index",
    "Temperature_Stress": "Aggregate Quality Index",
}


# ------------------------------------------------------------------
# STAGE METADATA (PLACEHOLDER - EDIT WITH REAL FAB DOMAIN KNOWLEDGE)
# ------------------------------------------------------------------
# disposition options: "rework", "material_recycle", "scrap", "inspect_only"
STAGE_METADATA = {
    "Crystal Growth": {
        "reworkable": False,
        "disposition": "material_recycle",
        "action": "Do not process further. Route ingot/wafer for silicon material reclaim.",
        "precaution": (
            "Crystal growth defects (dislocations, resistivity drift) are baked "
            "into the ingot and cannot be corrected downstream. Recommend "
            "reviewing furnace temperature and pull-rate control loops for this "
            "growth run."
        ),
    },
    "Wafer Slicing": {
        "reworkable": False,
        "disposition": "material_recycle",
        "action": "Scrap this wafer slice; reclaim silicon material where possible.",
        "precaution": (
            "Slicing defects (cracks, thickness variation) are physical and "
            "irreversible. Check blade speed/feed rate calibration and vibration "
            "isolation on the saw."
        ),
    },
    "Initial Polishing": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Send back for re-polishing before proceeding.",
        "precaution": (
            "Surface roughness/flatness issues from polishing can often be "
            "corrected with an additional polish pass. Verify pad condition and "
            "slurry flow rate."
        ),
    },
    "Cleaning": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Recirculate wafer through cleaning step.",
        "precaution": (
            "Particle contamination from cleaning is usually correctable with a "
            "repeat clean cycle. Check chemical concentration and cleaning time."
        ),
    },
    "Oxidation": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Strip oxide layer and re-run oxidation with corrected parameters.",
        "precaution": (
            "Oxide thickness/uniformity issues can typically be stripped (wet "
            "etch) and regrown. Review furnace temperature uniformity and O2 "
            "flow control."
        ),
    },
    "Photoresist Coating": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Strip resist and recoat.",
        "precaution": (
            "Resist thickness or bake issues are correctable by stripping and "
            "reapplying photoresist. Check spin speed consistency and bake "
            "plate temperature."
        ),
    },
    "Photolithography": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Strip resist and re-expose with corrected alignment/focus.",
        "precaution": (
            "Alignment or focus errors can usually be reworked by stripping "
            "resist and re-exposing, provided no etch has occurred yet. "
            "Recalibrate stepper alignment and focus offset."
        ),
    },
    "Etching": {
        "reworkable": False,
        "disposition": "scrap",
        "action": "Hold for manual engineering review; likely scrap.",
        "precaution": (
            "Etching is a subtractive, largely irreversible step. Critical "
            "dimension or etch-rate deviations here are difficult to correct "
            "post-hoc. Review RF power and gas flow (SF6/CF4/O2) stability."
        ),
    },
    "Ion Implantation": {
        "reworkable": False,
        "disposition": "scrap",
        "action": "Hold for manual engineering review; likely scrap.",
        "precaution": (
            "Dopant implantation is permanent and cannot be undone. Deviations "
            "in dose/energy require investigating beam current stability and "
            "implant energy calibration for future lots."
        ),
    },
    "Thin Film Deposition": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Strip deposited film and redeposit with corrected parameters.",
        "precaution": (
            "Film thickness deviations can often be stripped and redeposited "
            "if caught early. Check chamber pressure and gas flow consistency."
        ),
    },
    "Final CMP": {
        "reworkable": True,
        "disposition": "rework",
        "action": "Limited rework possible: light re-polish, monitor over-thinning risk.",
        "precaution": (
            "Final CMP flatness issues allow limited rework, but repeated "
            "polishing risks over-thinning the film stack. Recommend a single "
            "corrective pass with tightened down-force control."
        ),
    },
    "Final Metrology": {
        "reworkable": False,
        "disposition": "inspect_only",
        "action": "This is a detection point, not a root cause - trace back to the earlier stage that triggered it.",
        "precaution": (
            "Final metrology features (particle count, defect count) reflect "
            "accumulated upstream issues. Use the other contributing stages in "
            "this explanation to find the true root cause."
        ),
    },
    "Aggregate Quality Index": {
        "reworkable": False,
        "disposition": "inspect_only",
        "action": "These are summary indices, not a process step - use other contributing features to find the actionable root cause.",
        "precaution": (
            "Aggregate scores (Quality_Score, Process_Error_Index, etc.) "
            "summarize overall wafer health. Refer to the specific stage "
            "features also flagged in this explanation for a concrete action."
        ),
    },
}