# Drone SAR System Performance Audit Report

## Overall Acceptance Verdict: **PASSED**

### 1. Key Performance Criteria
| Metric | Acceptance Threshold | Measured Result | Verdict |
| :--- | :--- | :--- | :--- |
| **Recall @ IoU 0.5** | $\ge 90.0\%$ | **100.0%** | PASS |
| **Mean Pipeline Latency** | $< 300.0\text{ ms}$ | **31.4 ms** | PASS |
| **Deduplication Accuracy** | $\ge 85.0\%$ | **100.0%** | PASS |
| **False Positives / Min** | $< 10.0\text{ FP/min}$ | **0.00 FP/min** | PASS |

---

### 2. Operational Safety Guardrail
- **Safety Policy**: ENFORCED (Recommender Only - Zero Auto-Clearing Permitted)
- **Enforcement Detail**: Software strictly issues prioritized candidate triage advisories. Zero search grids or sectors are cleared automatically without explicit human SAR Commander authorization.

---

### 3. Detailed Statistics
- **Total Ground Truth Instances**: 110
- **True Positives**: 110
- **False Positives**: 0
- **False Negatives**: 0
- **Precision @ 0.5**: 100.0%
- **F1 Score**: 1.000
- **95th Percentile Latency**: 39.3 ms
- **Max Latency**: 80.7 ms
- **Frames Evaluated**: 60 (2.0s)
