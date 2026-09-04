"""
Generates a professional academic Word Document (.docx) report for Digital Assignment 1 (DA-1).
Course: DBMS BCSE302L
Title: Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting in Distributed Solar Rooftop Systems
Authors: Revanth P. A. (24BCE1730) & Siripuram Srimannath (24BCE5034)
"""

import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DOCX = PROJECT_ROOT / "DA1_Project_Report_Solar_Forecasting.docx"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def set_cell_background(cell, hex_color: str):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_heading_styled(doc, text, level=1):
    """Adds styled headings with professional colors."""
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    for run in h.runs:
        if level == 1:
            run.font.color.rgb = RGBColor(15, 23, 42)  # Slate 900
            run.font.name = "Calibri"
            run.font.size = Pt(16)
            run.bold = True
        elif level == 2:
            run.font.color.rgb = RGBColor(30, 58, 138)  # Deep Navy
            run.font.name = "Calibri"
            run.font.size = Pt(13)
            run.bold = True
        elif level == 3:
            run.font.color.rgb = RGBColor(51, 65, 85)
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.bold = True
    return h


def create_report():
    doc = Document()

    # Set Standard Page Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Style Configurations
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # ----------------------------------------------------
    # Title Section
    # ----------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run(
        "Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting\n"
        "in Distributed Solar Rooftop Systems"
    )
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    # Subtitle / Course Banner
    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Course: DBMS BCSE302L — Digital Assignment 1 (DA-1) | Project Report")
    r_sub.font.size = Pt(12)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(37, 99, 235)  # Blue 600

    # Authors Table Box
    auth_table = doc.add_table(rows=1, cols=1)
    auth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = auth_table.cell(0, 0)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    p_auth = cell.paragraphs[0]
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(0)
    r_a = p_auth.add_run(
        "Project Authors & System Architecture Ownership:\n"
        "Revanth P. A. (Registration No: 24BCE1730)\n"
        "Siripuram Srimannath (Registration No: 24BCE5034)\n"
        "School of Computer Science and Engineering, Vellore Institute of Technology (VIT)\n"
        "GitHub Repository: https://github.com/opafkkabaap/solar-hyperlocal-forecasting"
    )
    r_a.font.size = Pt(10)
    r_a.font.italic = True
    r_a.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ----------------------------------------------------
    # Abstract
    # ----------------------------------------------------
    add_heading_styled(doc, "Executive Abstract", level=1)
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.add_run(
        "Distributed solar photovoltaic (PV) rooftop installations operate in environments subject to severe microclimatic volatility, "
        "rapid cloud shadow transients, and localized thermal shifts. Standard Numerical Weather Prediction (NWP) models operate across coarse spatial "
        "grids (10–50 km) and multi-hour update frequencies, rendering them ill-equipped for short-term photovoltaic yield forecasting at the individual "
        "building level. This project constructs an end-to-end operational software and deep learning framework that ingests heterogeneous multi-modal data "
        "streams—combining on-site ground telemetry (pyranometer irradiance, ambient surface temperature, PV module back-of-panel temperature) with satellite "
        "reanalysis meteorological feeds fetched dynamically via the NASA POWER REST API for the Pune metropolitan region (18.5204° N, 73.8567° E).\n\n"
        "The automated data engineering pipeline executes physical boundary enforcement, unsupervised Isolation Forest anomaly detection, and cubic spline "
        "temporal interpolation from 1-hour feeds down to uniform 15-minute intervals. To eliminate documented sensor logging discrepancies, grid-deliverable AC power "
        "(total_ac_power) is established as the sole target variable across 13 multi-modal features. We implement, train, and benchmark two sequential recurrent "
        "deep learning architectures—Long Short-Term Memory (SolarLSTM) and Gated Recurrent Unit (SolarGRU)—for multi-step ultra-short-term forecasting (H = 4 steps, "
        "predicting 15, 30, 45, and 60 minutes ahead). Evaluated strictly on an unseen 5-day chronological test partition (June 13–17, 2020), SolarLSTM achieves an overall "
        "R² score of 0.9034 (RMSE = 2,542.68 kW), while SolarGRU achieves a competitive R² of 0.8991 (RMSE = 2,598.91 kW) with 24.0% fewer trainable parameters (42,340 vs. 55,716) "
        "and faster convergence, proving ideal for edge-node microgrid controllers."
    )

    p_kw = doc.add_paragraph()
    r_kwt = p_kw.add_run("Keywords: ")
    r_kwt.bold = True
    p_kw.add_run("Multi-Modal AI, Hyperlocal Weather Forecasting, Distributed Solar Rooftop Systems, TimescaleDB, LSTM, GRU, Isolation Forest, NASA POWER API.")

    # ----------------------------------------------------
    # 1. System Overview & Core Dimensions
    # ----------------------------------------------------
    add_heading_styled(doc, "1. System Architecture & Core Dimensions", level=1)
    doc.add_paragraph(
        "The proposed framework bridges heterogeneous meteorological observation channels, time-series relational database infrastructure, "
        "recurrent neural sequence models, and physics-informed photovoltaic power conversion equations into a unified pipeline."
    )

    # Core System Dimension Table
    dim_table = doc.add_table(rows=7, cols=3)
    dim_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Core System Dimension", "Architectural Specification", "Technical Function & Role"]
    for j, h_text in enumerate(headers):
        c = dim_table.cell(0, j)
        set_cell_background(c, "1E293B")
        set_cell_margins(c, top=120, bottom=120, left=150, right=150)
        p = c.paragraphs[0]
        r = p.add_run(h_text)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    dim_data = [
        ("Course & Project Module", "DBMS BCSE302L — Digital Assignment 1 (DA-1)", "Academic governance, evaluation baseline, and delivery scope."),
        ("Primary Target Output", "Grid-Deliverable AC Power Output (total_ac_power in kW)", "Direct multi-step short-term nowcasting (15, 30, 45, 60 min ahead)."),
        ("Ingestion Modalities", "Ground Pyranometer Telemetry + NASA POWER REST API", "Multi-modal fusion of localized sensors with satellite atmospheric reanalysis."),
        ("Geographic Baseline", "Pune Metropolitan Region (18.5204° N, 73.8567° E)", "Synchronizes celestial sun angles, solar noon, and microclimatic parameters."),
        ("Sequential AI Models", "SolarLSTM (55.7k Params) & SolarGRU (42.3k Params)", "Comparative temporal modeling under identical sequence windows."),
        ("Statistical Benchmarks", "RMSE (kW), MAE (kW), Daytime MAPE (%), R² Score", "Standardized empirical evaluation on unseen chronological test partition."),
    ]

    for i, row in enumerate(dim_data, start=1):
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            c = dim_table.cell(i, j)
            set_cell_background(c, bg)
            set_cell_margins(c, top=100, bottom=100, left=150, right=150)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ----------------------------------------------------
    # 2. Database Management & Schema Design
    # ----------------------------------------------------
    add_heading_styled(doc, "2. Database Management & Relational Schema Design", level=1)
    doc.add_paragraph(
        "To fulfill the database engineering requirements of DBMS BCSE302L, the persistent storage architecture is structured using PostgreSQL "
        "with time-series hyper-table indexing. Multi-modal observations are indexed by timestamp and station identifier, enforcing data integrity constraints."
    )

    # SQL Schema Box
    sql_table = doc.add_table(rows=1, cols=1)
    sql_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_sql = sql_table.cell(0, 0)
    set_cell_background(c_sql, "F8FAFC")
    set_cell_margins(c_sql, top=120, bottom=120, left=150, right=150)
    p_sql = c_sql.paragraphs[0]
    p_sql.paragraph_format.space_after = Pt(0)
    r_code = p_sql.add_run(
        "-- Weather Station Master Table\n"
        "CREATE TABLE WeatherStation (\n"
        "    station_id VARCHAR(50) PRIMARY KEY,\n"
        "    name VARCHAR(100) NOT NULL,\n"
        "    latitude NUMERIC(9,6) NOT NULL,\n"
        "    longitude NUMERIC(9,6) NOT NULL,\n"
        "    elevation NUMERIC(6,2)\n"
        ");\n\n"
        "-- Multi-Modal Time-Series Hyper-table\n"
        "CREATE TABLE MultiModalSolarData (\n"
        "    timestamp TIMESTAMPTZ NOT NULL,\n"
        "    station_id VARCHAR(50) REFERENCES WeatherStation(station_id),\n"
        "    total_ac_power NUMERIC(10,3) CHECK (total_ac_power >= 0),\n"
        "    sensor_irradiation NUMERIC(6,3) CHECK (sensor_irradiation >= 0),\n"
        "    sensor_ambient_temp NUMERIC(5,2),\n"
        "    sensor_module_temp NUMERIC(5,2),\n"
        "    nasa_ghi NUMERIC(7,2) CHECK (nasa_ghi >= 0),\n"
        "    nasa_cloud_amt NUMERIC(5,2) CHECK (nasa_cloud_amt BETWEEN 0 AND 100),\n"
        "    nasa_ambient_temp NUMERIC(5,2),\n"
        "    nasa_rh NUMERIC(5,2) CHECK (nasa_rh BETWEEN 0 AND 100),\n"
        "    nasa_wind_speed NUMERIC(5,2) CHECK (nasa_wind_speed >= 0),\n"
        "    nasa_pressure NUMERIC(6,2),\n"
        "    nasa_clearness_index NUMERIC(4,3),\n"
        "    is_anomaly SMALLINT DEFAULT 0,\n"
        "    PRIMARY KEY (timestamp, station_id)\n"
        ");"
    )
    r_code.font.name = "Consolas"
    r_code.font.size = Pt(8.5)
    r_code.font.color.rgb = RGBColor(15, 23, 42)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ----------------------------------------------------
    # 3. Data Cleaning, Sanitization & Feature Engineering
    # ----------------------------------------------------
    add_heading_styled(doc, "3. Data Cleaning, Sanitization & Feature Engineering", level=1)
    
    add_heading_styled(doc, "3.1 Resolving Plant 1 DC Scale Quirk: AC Power Formulation", level=2)
    doc.add_paragraph(
        "A critical finding during exploratory data analysis of the Kaggle ground-truth dataset was an anomaly in Plant 1's DC power readings, "
        "which showed AC/DC ratios ≈ 0.098 (a known logging unit inconsistency where raw DC values were logged in tens or per-string units relative to Plant 2's ~0.977). "
        "To maintain rigorous physical validity without applying speculative ad-hoc multipliers, grid-deliverable AC power (total_ac_power) was established "
        "as the sole forecasting target (y). DC_POWER was dropped from input features and target formulation, preventing models from learning spurious plant-specific scale artifacts."
    )

    add_heading_styled(doc, "3.2 Unsupervised Isolation Forest Anomaly Smoothing", level=2)
    doc.add_paragraph(
        "An Isolation Forest algorithm was deployed on multi-variate telemetry features (AC power, on-site irradiation, sensor ambient temperature, NASA GHI), "
        "flagging 33 outlier intervals (1.01%) caused by transient telemetry dropouts. In continuous time-series sequence windowing, dropping rows breaks "
        "the temporal step Δt = 15 min. Therefore, flagged anomalies were smoothed via local rolling linear interpolation, preserving continuity while eliminating telemetry spikes."
    )

    add_heading_styled(doc, "3.3 Diurnal Zero-Inflation & Cyclical Encodings", level=2)
    doc.add_paragraph(
        "Nocturnal observations (PAC = 0, GHI = 0) were intentionally retained to allow recurrent neural networks to learn the 24-hour diurnal solar cycle. "
        "To enable neural networks to perceive continuous time across midnight boundaries, continuous trigonometric features were engineered:\n"
        "    hour_sin = sin(2π · t_hour / 24),    hour_cos = cos(2π · t_hour / 24)"
    )

    add_heading_styled(doc, "3.4 Leakage-Free Normalization & Chronological Partition", level=2)
    doc.add_paragraph(
        "Data leakage was strictly prevented by partitioning chronologically without random shuffling:\n"
        "• Train Partition: First 24 days (May 15 to June 7, 2020) — 2,205 sequence windows (70.6%)\n"
        "• Validation Partition: Next 5 days (June 8 to June 12, 2020) — 477 sequence windows (14.7%)\n"
        "• Test Partition: Last 5 days (June 13 to June 17, 2020) — 477 sequence windows (14.7%)\n\n"
        "A custom LeakFreeMinMaxScaler was fitted strictly on the Train partition and applied to validation and test sets. "
        "Inverse transform precision tests verified an exact reconstruction error of 0.00e+00."
    )

    # ----------------------------------------------------
    # 4. Deep Learning Modeling Architectures
    # ----------------------------------------------------
    add_heading_styled(doc, "4. Deep Learning Sequence Modeling Architectures", level=1)
    doc.add_paragraph(
        "Both SolarLSTM and SolarGRU architectures process 3D sequence tensors of shape (Batch, 96, 13)—representing 24 hours of 15-minute multi-modal history—"
        "to output 4-step ahead multi-step nowcasts (Batch, 4) corresponding to predictions at t+15m, t+30m, t+45m, and t+60m."
    )

    # Mathematical Formulations Table
    add_heading_styled(doc, "4.1 Mathematical Formulations & Gating Equations", level=2)
    math_table = doc.add_table(rows=3, cols=2)
    math_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    math_table.cell(0, 0).paragraphs[0].add_run("SolarLSTM (Long Short-Term Memory)").bold = True
    math_table.cell(0, 1).paragraphs[0].add_run("SolarGRU (Gated Recurrent Unit)").bold = True
    set_cell_background(math_table.cell(0, 0), "1E293B")
    set_cell_background(math_table.cell(0, 1), "1E293B")
    math_table.cell(0, 0).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    math_table.cell(0, 1).paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    lstm_math = (
        "Forget Gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)\n"
        "Input Gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)\n"
        "Candidate:    C~_t = tanh(W_c · [h_{t-1}, x_t] + b_c)\n"
        "Cell State:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t\n"
        "Output Gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)\n"
        "Hidden State: h_t = o_t ⊙ tanh(C_t)"
    )
    gru_math = (
        "Update Gate:  z_t = σ(W_z · [h_{t-1}, x_t] + b_z)\n"
        "Reset Gate:   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)\n"
        "Candidate:    h~_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)\n"
        "Hidden State: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t\n\n"
        "Eliminates separate cell state C_t;\n"
        "Reduces parameter count by 24.0%."
    )

    c0 = math_table.cell(1, 0)
    c1 = math_table.cell(1, 1)
    set_cell_background(c0, "F8FAFC")
    set_cell_background(c1, "F8FAFC")
    set_cell_margins(c0, top=100, bottom=100, left=120, right=120)
    set_cell_margins(c1, top=100, bottom=100, left=120, right=120)
    
    r0 = c0.paragraphs[0].add_run(lstm_math)
    r0.font.name = "Consolas"
    r0.font.size = Pt(8.5)
    r1 = c1.paragraphs[0].add_run(gru_math)
    r1.font.name = "Consolas"
    r1.font.size = Pt(8.5)

    c0_p = math_table.cell(2, 0)
    c1_p = math_table.cell(2, 1)
    set_cell_background(c0_p, "FFFFFF")
    set_cell_background(c1_p, "FFFFFF")
    c0_p.paragraphs[0].add_run("Trainable Parameters: 55,716\nBest Val Loss: 0.006390 (Epoch 31)").font.size = Pt(9)
    c1_p.paragraphs[0].add_run("Trainable Parameters: 42,340 (-24.0%)\nBest Val Loss: 0.006055 (Epoch 26)").font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ----------------------------------------------------
    # 5. Empirical Experimental Results
    # ----------------------------------------------------
    add_heading_styled(doc, "5. Empirical Experimental Results & Comparative Benchmarks", level=1)
    doc.add_paragraph(
        "Both architectures were trained under identical conditions (Adam optimizer lr=0.001, ReduceLROnPlateau factor=0.5, patience=5, early stopping patience=10) "
        "and evaluated on the unseen test partition (June 13–17, 2020). Daytime MAPE is computed on active daylight generation using the data-derived "
        "threshold y > 0.01 · y_max (278.95 kW), eliminating nocturnal zero distortions without arbitrary hardcoded cutoffs."
    )

    # Table 5.1: Overall Benchmarks
    add_heading_styled(doc, "5.1 Overall Horizon Comparative Benchmarks (1-Hour Ahead)", level=2)
    t_ov = doc.add_table(rows=3, cols=7)
    t_ov.alignment = WD_TABLE_ALIGNMENT.CENTER
    ov_headers = ["Model", "Parameters", "Best Val Loss", "Overall R²", "Overall RMSE (kW)", "Overall MAE (kW)", "Daytime MAPE (%)"]
    for j, h in enumerate(ov_headers):
        c = t_ov.cell(0, j)
        set_cell_background(c, "1E293B")
        set_cell_margins(c, top=100, bottom=100, left=100, right=100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9)

    ov_rows = [
        ("SolarLSTM", "55,716", "0.006390", "0.9034", "2,542.68", "1,377.94", "31.00%"),
        ("SolarGRU", "42,340 (-24%)", "0.006055", "0.8991", "2,598.91", "1,433.31", "30.53%"),
    ]
    for i, row in enumerate(ov_rows, start=1):
        bg = "F8FAFC" if i == 1 else "FFFFFF"
        for j, val in enumerate(row):
            c = t_ov.cell(i, j)
            set_cell_background(c, bg)
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j in [3, 4, 5, 6] and ((i == 1 and j in [3, 4, 5]) or (i == 2 and j == 6)):
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Table 5.2: Per-Step Breakdown
    add_heading_styled(doc, "5.2 Step-by-Step Lead-Time Progression (t+15m to t+60m)", level=2)
    t_step = doc.add_table(rows=5, cols=9)
    t_step.alignment = WD_TABLE_ALIGNMENT.CENTER
    step_headers = ["Lead Time", "LSTM R²", "GRU R²", "LSTM RMSE", "GRU RMSE", "LSTM MAE", "GRU MAE", "LSTM MAPE", "GRU MAPE"]
    for j, h in enumerate(step_headers):
        c = t_step.cell(0, j)
        set_cell_background(c, "1E293B")
        set_cell_margins(c, top=100, bottom=100, left=80, right=80)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(8.5)

    step_rows = [
        ("t+15 min", "0.9221", "0.9219", "2,283.87 kW", "2,286.97 kW", "1,232.84 kW", "1,253.54 kW", "27.72%", "25.44%"),
        ("t+30 min", "0.9075", "0.9034", "2,488.26 kW", "2,542.15 kW", "1,325.16 kW", "1,400.31 kW", "29.56%", "29.53%"),
        ("t+45 min", "0.8972", "0.8917", "2,623.55 kW", "2,692.60 kW", "1,430.87 kW", "1,474.74 kW", "32.61%", "32.45%"),
        ("t+60 min", "0.8869", "0.8794", "2,751.45 kW", "2,841.56 kW", "1,522.89 kW", "1,604.64 kW", "34.10%", "34.70%"),
    ]
    for i, row in enumerate(step_rows, start=1):
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            c = t_step.cell(i, j)
            set_cell_background(c, bg)
            set_cell_margins(c, top=80, bottom=80, left=80, right=80)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ----------------------------------------------------
    # 6. Visual Performance & Embedded Figures
    # ----------------------------------------------------
    add_heading_styled(doc, "6. Visual Performance & Generated Figures", level=1)

    # Loss Curves
    add_heading_styled(doc, "6.1 Training & Validation Convergence Curves", level=2)
    p_loss_desc = doc.add_paragraph(
        "Figure 1 and Figure 2 display the training and validation loss trajectories across epochs for SolarLSTM and SolarGRU. "
        "Both models exhibit smooth convergence without divergence, with SolarGRU triggering early stopping at epoch 26 (val loss: 0.006055)."
    )

    lstm_loss_img = FIGURES_DIR / "lstm_plant1_loss.png"
    gru_loss_img = FIGURES_DIR / "gru_plant1_loss.png"

    if lstm_loss_img.exists():
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        doc.add_picture(str(lstm_loss_img), width=Inches(5.8))
        p_cap1 = doc.add_paragraph("Figure 1: SolarLSTM Training and Validation Loss Curve across 40 Epochs.")
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.runs[0].font.size = Pt(9)
        p_cap1.runs[0].font.italic = True

    if gru_loss_img.exists():
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        doc.add_picture(str(gru_loss_img), width=Inches(5.8))
        p_cap2 = doc.add_paragraph("Figure 2: SolarGRU Training and Validation Loss Curve with Early Stopping at Epoch 26.")
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.runs[0].font.size = Pt(9)
        p_cap2.runs[0].font.italic = True

    # Forecast vs Actual
    add_heading_styled(doc, "6.2 Five-Day Unseen Test Partition Forecast vs. Actual Curves", level=2)
    doc.add_paragraph(
        "Figure 3 and Figure 4 illustrate the 5-day continuous test sequence (June 13 to June 17, 2020) comparing ground-truth photovoltaic AC power generation "
        "against the immediate 15-minute (t+15m) and 1-hour ahead (t+60m) predictions. The models capture diurnal sunrise ramps, solar noon peaks, and evening ramp-downs."
    )

    lstm_fc_img = FIGURES_DIR / "lstm_plant1_forecast.png"
    gru_fc_img = FIGURES_DIR / "gru_plant1_forecast.png"

    if lstm_fc_img.exists():
        doc.add_picture(str(lstm_fc_img), width=Inches(6.2))
        p_cap3 = doc.add_paragraph("Figure 3: SolarLSTM 1-Hour Ahead Multi-Step Forecast vs. Actual Generation across 5-Day Test Partition.")
        p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap3.runs[0].font.size = Pt(9)
        p_cap3.runs[0].font.italic = True

    if gru_fc_img.exists():
        doc.add_picture(str(gru_fc_img), width=Inches(6.2))
        p_cap4 = doc.add_paragraph("Figure 4: SolarGRU 1-Hour Ahead Multi-Step Forecast vs. Actual Generation across 5-Day Test Partition.")
        p_cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap4.runs[0].font.size = Pt(9)
        p_cap4.runs[0].font.italic = True

    # ----------------------------------------------------
    # 7. Physics-Informed Solar Power Conversion Engine
    # ----------------------------------------------------
    add_heading_styled(doc, "7. Physics-Informed Photovoltaic Conversion Engine", level=1)
    doc.add_paragraph(
        "To translate forecasted meteorological metrics (GHI and ambient temperature) into expected photovoltaic kilowatt yield, "
        "the architecture couples sequence model outputs with physical solar conversion equations:"
    )

    p_pv = doc.add_paragraph()
    p_pv.paragraph_format.line_spacing = 1.2
    p_pv.add_run(
        "1. PV Cell Temperature Equation:\n"
        "   T_cell = T_ambient + (GHI / 800) · (NOCT - 20)\n"
        "   Where NOCT = 45°C represents Nominal Operating Cell Temperature.\n\n"
        "2. Dynamic Temperature-Dependent Efficiency:\n"
        "   η_cell = η_STC · [1 + γ · (T_cell - 25)]\n"
        "   Where η_STC = 18.5% is baseline efficiency and γ = -0.0038 / °C is the thermal degradation coefficient.\n\n"
        "3. Total System Expected Power Generation:\n"
        "   P_PV = A_array · GHI · η_cell · PR\n"
        "   Where PR ≈ 0.82 is the balance-of-system Performance Ratio (accounting for inverter losses and wiring resistance)."
    )
    p_pv.runs[0].font.name = "Consolas"
    p_pv.runs[0].font.size = Pt(9.5)

    # ----------------------------------------------------
    # 8. Discussion & Key Academic Insights
    # ----------------------------------------------------
    add_heading_styled(doc, "8. Discussion & Academic Insights for DA-1", level=1)
    doc.add_paragraph(
        "1. Computational vs. Accuracy Trade-off:\n"
        "   SolarLSTM achieves slightly higher variance capture at the 60-minute mark (R² = 0.8869 vs 0.8794) due to its separate cell memory state (C_t), "
        "   making it suitable for centralized grid scheduling. SolarGRU achieves virtually identical immediate accuracy (R² = 0.9219 at t+15m) with 24.0% fewer "
        "   trainable parameters (42,340 vs 55,716) and faster training convergence (26 epochs), making it optimal for resource-constrained edge microgrid units.\n\n"
        "2. Monotonic Horizon Error Progression:\n"
        "   The steady degradation of R² from 0.922 at 15 minutes to 0.887 at 60 minutes reflects genuine physical atmospheric volatility. Multi-modal fusion "
        "   (satellite NASA GHI + on-site pyranometer) provides rich immediate boundary conditions, while physical uncertainty compounds with lead time.\n\n"
        "3. Defensible Evaluation Metrics:\n"
        "   Using the data-derived threshold y > 0.01 · y_max (278.95 kW) successfully eliminates nocturnal zero-denominator distortion in MAPE calculations "
        "   without resorting to arbitrary hardcoded numbers."
    )

    # ----------------------------------------------------
    # 9. Formal References
    # ----------------------------------------------------
    add_heading_styled(doc, "9. References & Dataset Citations", level=1)
    refs = [
        "[1] A. Kannal, 'Solar Power Generation Data: Telemetry and Weather Sensor Readings from Two Solar Power Plants in India,' Kaggle Datasets, 2020. [Online]. Available: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data.",
        "[2] NASA Langley Research Center, 'Prediction of Worldwide Energy Resources (POWER) Hourly Point Meteorological and Solar Data,' NASA POWER Project, 2026. [Online]. Available: https://power.larc.nasa.gov.",
        "[3] X. Su et al., 'Dynamic Hierarchical Graph Convolutional Network for Distributed Solar Forecasting,' IEEE Transactions on Sustainable Energy, 2025.",
        "[4] Y. Wu et al., 'Mixed Kernel Support Vector Machine with Isolation Forest Anomaly Detection for Photovoltaic Yield Estimation,' IEEE Access, 2025.",
        "[5] Z. Wang et al., 'Satellite Image Cloud Tracking Model for Solar Power Nowcasting across Distributed Nodes,' IEEE Transactions on Geoscience and Remote Sensing, 2022."
    ]
    for ref_text in refs:
        p_ref = doc.add_paragraph(ref_text)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.runs[0].font.size = Pt(9)

    doc.save(OUTPUT_DOCX)
    print(f"Successfully generated academic Word Report: {OUTPUT_DOCX.resolve()}")


if __name__ == "__main__":
    create_report()
