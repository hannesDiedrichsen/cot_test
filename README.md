# Zero Shot Chain-of-Thought Benchmark

Benchmark-System zur Evaluierung der Zero Shot Chain-of-Thought (CoT) Fähigkeiten von Gemini-Modellen.

## Features

- **Zweistufiges Prompt-System**: Erst Reasoning, dann standardisiertes Follow-up für finale Antwort
- **15+ Datensätze**: Grundlegende + erweiterte Mathematik/Logik-Datasets
- **2 Methoden**: Zero Shot vs. Zero Shot CoT ("Let's think step by step")
- **Gemini-Modelle**: gemini-2.5-flash, gemini-1.5-pro
- **Automatische Visualisierung**: Vergleichsdiagramme nach jedem Benchmark
- **Detailliertes Logging**: Vollständige Nachverfolgung beider Prompt-Stufen

## Verfügbare Datasets

### **Basis-Datasets:**
- AddSub, CommonsenseQA, MultiArith, SingleEq, coin_flip, grade-school-math, last_letters

### **Erweiterte Mathematik:**
- **MATH** - Competition Mathematics (sehr schwer)  
- **GSM8K** - Grade School Math (erweitert)
- **MathQA** - Math Word Problems (Multiple Choice)
- **TheoremQA** - Theorem-basierte Probleme

### **Logik & Reasoning:**
- **LogiQA** - Logical Reasoning Questions
- **ReClor** - Reading Comprehension + Logic
- **ProntoQA** - Synthetic Logic Problems  
- **AR-LSAT** - LSAT Logical Reasoning

## Setup

1. **Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

2. **API Key konfigurieren:**
```bash
cp .env.example .env
```

3. **Testen der Installation:**
```bash
python main.py --datasets AddSub --sample-size 5 --models gemini-2.5-flash --methods zero_shot
```

## Verwendung

### Vollständiger Benchmark (alle Datensätze):
```bash
python main.py
```

### Spezifische Datensätze:
```bash
python main.py --datasets AddSub MultiArith --sample-size 10
```

### Nur Zero Shot CoT:
```bash
python main.py --methods zero_shot_cot --models gemini-1.5-pro
```

### Mit deaktiviertem Denken:
```bash
# Gemini's internes "Denken" ausschalten
python main.py --datasets GSM8K --no-thinking --sample-size 10

# Custom Thinking Budget
python main.py --datasets ARC_Challenge --thinking-budget 5000 --sample-size 10
```

### Verfügbare Parameter:
- `--datasets`: AddSub, CommonsenseQA, MultiArith, SingleEq, coin_flip, grade-school-math, last_letters, GSM8K, ARC_Challenge, BoolQ, Winogrande
- `--models`: gemini-2.5-flash, gemini-1.5-pro  
- `--methods`: zero_shot, zero_shot_cot
- `--sample-size`: Anzahl Fragen pro Dataset (Standard: alle)
- `--no-thinking`: Deaktiviert Geminis internes "Denken" 
- `--thinking-budget N`: Setzt Custom Thinking Budget (0=aus, >0=limitiert)

## Outputs

- **Detaillierte Ergebnisse**: `results/benchmark_results_TIMESTAMP.json`
- **CSV Summary**: `results/benchmark_summary_TIMESTAMP.csv`  
- **Vergleichsdiagramme**: `results/benchmark_chart_TIMESTAMP.png`
- **Multi-Model Charts**: `results/benchmark_chart_TIMESTAMP_models.png` (bei mehreren Modellen)
- **Logs**: `logs/benchmark_results.log`

## Beispiel Output

```
BENCHMARK SUMMARY
================
Overall Accuracy: 0.847 (245/289)

Accuracy by Method:
  zero_shot: 0.812 (117/144)
  zero_shot_cot: 0.883 (128/145)

Accuracy by Dataset:
  AddSub: 0.892 (33/37)
  CommonsenseQA: 0.756 (31/41)
  grade-school-math: 0.923 (36/39)
```

## Nachträgliche Chart-Generierung

### Einzelne Datei:
```bash
# Aus JSON Ergebnissen
python generate_charts.py results/benchmark_results_20250729_125508.json

# Aus CSV Summary
python generate_charts.py results/benchmark_summary_20250729_125508.csv

# Neueste Ergebnisse aus Ordner
python generate_charts.py --input results/ --latest
```

### Batch-Verarbeitung:
```bash
# Alle Benchmark-Dateien im results/ Ordner
python batch_generate_charts.py

# Custom Ordner
python batch_generate_charts.py --directory data/benchmarks/

# Mit custom Output-Ordner
python batch_generate_charts.py --output charts/
```

## Konfiguration

Anpassungen in `config/config.yaml`:
- Rate Limiting
- Timeout-Werte  
- Log-Level
- Dataset-Pfade