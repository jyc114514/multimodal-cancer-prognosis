# Project Background

The broader undergraduate research direction was retrospective TCGA-BRCA cancer-prognosis modeling with whole-slide-derived pathology features, mRNA expression and clinical information. Across historical research versions, the work explored alternative UNI/CONCH pathology representations, attention-based MIL, multimodal fusion, Cox-style survival objectives and privileged clinical information.

The historical materials contain multiple model generations, and the original thesis-final artifact is not included in this repository. The exact mapping between every historical architecture and the submitted thesis configuration should not be inferred from this summary. Historical results and architecture diagrams are deliberately not reproduced here.

The current public repository does not attempt to reconstruct the full historical thesis architecture. It is a clean, reusable methodology reference centered on subject-level data handling, grouped splitting, optional fold-local variance selection, training-fold-only preprocessing, a simple Cox survival-risk model and censoring-aware evaluation.

The current public code does not implement the historical teacher/student distillation, privileged-stage path, HyperBank/HGNN, full MIL pipeline, gene-guided cross-modal attention or the exact Phase4 experiment snapshot. It reports no historical performance numbers and includes no study data.
