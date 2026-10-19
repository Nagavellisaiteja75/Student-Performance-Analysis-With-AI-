# Analysis-Focused Data Dictionary

The full UCI data description is available on the [dataset page](https://archive.ics.uci.edu/dataset/320/student%2Bperformance). This guide groups fields by how they are used in the project.

| Group | Fields | Meaning in this project |
| --- | --- | --- |
| Course context | `subject`, `school`, `reason`, `traveltime` | Course and school-choice context |
| Student / household context | `sex`, `age`, `address`, `famsize`, `Pstatus`, `guardian`, `Medu`, `Fedu`, `Mjob`, `Fjob` | Background information; do not treat as causal explanations |
| Study and support context | `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `higher`, `internet` | Learning-related and support-related context |
| Wellbeing and social context | `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `romantic`, `nursery` | Self-reported social and wellbeing context; use cautiously |
| Attendance context | `absences` | Must be captured at a defined point in time before prediction |
| Prior-grade fields | `G1`, `G2` | Excluded by default to avoid early-intervention leakage |
| Target | `G3` | Final course grade, 0–20 scale |

## Timing note

The source does not turn every field into an early-term feature automatically. In a real workflow, every field should carry an “available by” date. This portfolio project removes the most direct grade leakage (`G1` and `G2`) and prompts the dashboard user to enter only values known at the support point.

## Sensitive-use note

Even when a feature is technically available, it should not automatically be appropriate for intervention decisions. A real school would need a documented purpose, data-minimization review, student privacy safeguards, and fairness testing before using any demographic or social-context field operationally.
