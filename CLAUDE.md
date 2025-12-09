# CLAUDE.md - Development Context

This file provides context for AI coding assistants (like Claude Code) working on this project.

## Project Context

**Goal**: Build a computer vision pipeline for detecting behavioral and visual anomalies in underwater fish footage, demonstrating skills relevant to ML Engineer positions in computer vision and production ML systems.

**Target Company Context**: This project aligns with aquaculture technology companies using computer vision for fish health monitoring, weight estimation, and population management. The end product should show understanding of:
- Real-time processing constraints
- Robustness to challenging environmental conditions (underwater imaging)
- End-to-end pipeline thinking (data → model → deployment)
- Interpretable results for domain experts (fish farmers)

**Timeline**: This is a portfolio project with ~2-3 week target completion for MVP, with ongoing refinements.

## Developer Background

**Raymond Murray** - MS Data Science student (UT Austin, graduating April 2026) transitioning from quantitative neuroscience research to industry ML roles.

### Key Strengths to Leverage:
- **Time-series analysis**: 3+ years analyzing multi-modal neural and behavioral recordings
- **Feature engineering**: Successfully identified novel behavioral states through custom metric development
- **Statistical rigor**: Hypothesis testing, multiple comparison corrections, cross-validation
- **Production pipelines**: Built ETL workflows processing ~1,400 daily observations with real-time monitoring
- **Python/MATLAB**: Strong programming foundation, comfortable with data processing at scale

### Skills Being Developed:
- **Computer vision**: CV techniques are newer—provide clear explanations of CV concepts
- **PyTorch**: Has some experience but appreciates implementation details
- **Deployment**: Limited cloud/production deployment experience—include deployment considerations
- **SQL/Cloud platforms**: These are growth areas—flag opportunities to add these if relevant

### Working Preferences:
- **Actionable over theoretical**: Prefers concrete implementations with clear next steps
- **Asks clarifying questions**: Will push back if something doesn't make sense
- **Research-oriented**: Appreciates understanding the "why" behind technical choices
- **Documentation**: Values well-commented code and clear explanations

## Technical Specifications

### Core Technologies

**Required:**
- Python 3.9+
- PyTorch (primary DL framework)
- OpenCV (image processing)
- scikit-learn (classical ML, anomaly detection)
- pandas/NumPy (data manipulation)
- matplotlib/seaborn (visualization)

**Deployment/Interface:**
- Streamlit (web dashboard - prioritize this for portfolio visibility)
- Jupyter notebooks (exploratory analysis and documentation)

**Version Control:**
- Git (all development tracked)
- Clear commit messages following conventional commits style

### Dataset Information

**Primary**: DeepFish dataset (segmentation + classification of fish in marine environments)
- Location: https://alzayats.github.io/DeepFish/
- Contains: Annotated underwater fish images
- Challenges: Varied species, occlusions, water clarity issues

**Alternative/Supplementary**: 
- Fish4Knowledge dataset
- Brackish Underwater Dataset
- YouTube aquarium footage (for behavioral analysis)

**Note**: We'll likely need to supplement with video data since most datasets are image-focused, but behavioral anomalies require temporal analysis.

### Performance Requirements

**Not a real-time system initially**, but keep efficiency in mind:
- Target: Process 30 FPS video at ≥5 FPS (some latency acceptable for MVP)
- Memory: Should run on laptop GPU (if available) or CPU
- Scalability: Design to handle hours of footage, not just clips

## Development Phases

### Phase 1: Data Pipeline (Week 1)
**Status**: 🔄 In Progress

Tasks:
1. Download and organize DeepFish dataset
2. Build preprocessing pipeline:
   - Underwater image enhancement (color correction, contrast)
   - Frame extraction from video
   - Data augmentation for robustness
3. Exploratory data analysis notebook
4. Baseline statistics on image quality metrics

**Success Criteria**:
- Can load and visualize sample data
- Preprocessing improves visual quality (quantified with metrics like PSNR, SSIM)
- EDA notebook documents data distribution, challenges, and opportunities

### Phase 2: Detection & Tracking (Week 1-2)
**Status**: ⏳ Not Started

Tasks:
1. Implement fish detection (start with pretrained YOLO or Faster R-CNN)
2. Fine-tune detection model on DeepFish if needed
3. Add multi-object tracking (DeepSORT or ByteTrack)
4. Validate tracking accuracy on sample videos

**Success Criteria**:
- Detection mAP > 0.7 on test set
- Tracking maintains identity across occlusions
- Can process 10+ second clips end-to-end

### Phase 3: Feature Engineering (Week 2)
**Status**: ⏳ Not Started

**This is where Ray's research background shines—lean into it.**

Tasks:
1. Extract **behavioral features**:
   - Swimming speed (frame-to-frame displacement)
   - Trajectory smoothness (jerk, acceleration)
   - Spatial position distribution (clustering, wall proximity)
   - Activity level (movement variance over time windows)
2. Extract **visual features**:
   - Body posture (orientation, aspect ratio)
   - Color histograms (for appearance changes)
   - Texture features (potential lesions, fin damage)
   - Size estimates (bounding box area, aspect ratio)
3. Temporal aggregation:
   - Rolling statistics (mean, std, percentiles over N-second windows)
   - Frequency domain features (FFT of movement patterns)
4. Population-level features:
   - Individual deviation from group statistics
   - Comparative features (how does this fish differ from others?)

**Success Criteria**:
- Feature extraction pipeline outputs clean pandas DataFrame
- Features are normalized and validated (no NaNs, outliers documented)
- Notebook visualizes feature distributions and correlations

### Phase 4: Anomaly Detection Models (Week 2-3)
**Status**: ⏳ Not Started

Implement multiple approaches for comparison:

1. **Statistical baselines**:
   - Z-score threshold (simple but interpretable)
   - Isolation Forest (handles high-dimensional features well)
   - One-Class SVM (establish "normal" boundary)

2. **Deep learning**:
   - Autoencoder (reconstruction error for visual anomalies)
   - LSTM-Autoencoder (temporal behavioral patterns)
   - Optional: Variational Autoencoder for probabilistic anomalies

3. **Ensemble**:
   - Combine predictions from multiple models
   - Weighted voting or stacking

**Success Criteria**:
- At least 2 models trained and evaluated
- Comparison of precision/recall trade-offs
- Visualization of detected anomalies on sample videos
- Explainability: can identify which features triggered alerts

### Phase 5: Evaluation & Validation (Week 3)
**Status**: ⏳ Not Started

**Challenge**: Ground truth anomalies may not be labeled. Use proxy validation:

Tasks:
1. Synthetic anomalies:
   - Inject known abnormal behaviors (speed changes, erratic paths)
   - Test detection sensitivity
2. Statistical validation:
   - Anomaly score distributions
   - Temporal consistency (anomalies shouldn't flicker frame-to-frame)
3. Qualitative review:
   - Manual inspection of top-K anomalies
   - Document false positives vs. interesting catches
4. Performance benchmarking:
   - Processing speed (FPS)
   - Memory usage
   - Model size

**Success Criteria**:
- Can detect >80% of synthetic anomalies
- False positive rate < 10% (tunable threshold)
- Documentation of failure modes and limitations

### Phase 6: Dashboard & Deployment (Week 3)
**Status**: ⏳ Not Started

**Critical for portfolio**: Hiring managers need to see a working demo.

Tasks:
1. Streamlit app with:
   - Video upload interface
   - Real-time processing and visualization
   - Anomaly timeline/heatmap
   - Feature importance plots
   - Downloadable reports (CSV of anomalies)
2. Docker container (optional but valuable):
   - Reproducible environment
   - Shows deployment awareness
3. README updates:
   - Usage examples with screenshots
   - Demo GIF/video
4. Code cleanup:
   - Type hints
   - Docstrings
   - Unit tests for core functions (at least preprocessing and feature extraction)

**Success Criteria**:
- Non-technical user can run the app and understand results
- Deployed to GitHub with clear documentation
- Demo video shows end-to-end workflow

## Key Technical Decisions & Rationale

### Why Unsupervised Anomaly Detection?
- Labeled fish health data is scarce
- Anomaly definition varies by context (species, environment, feeding schedule)
- Unsupervised methods generalize better to new scenarios
- Demonstrates statistical thinking beyond supervised learning

### Why Combine Behavioral + Visual Features?
- Behavioral changes often precede visible symptoms
- Visual anomalies (lesions, discoloration) are easier to interpret
- Multi-modal approach is more robust
- Mirrors Ray's research: analyzing multiple data streams simultaneously

### Model Selection Philosophy:
- Start simple (statistical baselines) for interpretability
- Add complexity (deep learning) where justified by performance gains
- Prioritize explainability over marginal accuracy improvements
- Document trade-offs: speed vs. accuracy, simplicity vs. sophistication

## Code Quality Standards

### Structure:
- Modular design: Each phase (preprocessing, detection, features, models) is a separate module
- Config files (YAML) for hyperparameters and paths—avoid hardcoding
- Logging: Use Python `logging` module, not just print statements
- Error handling: Graceful failures with informative messages

### Documentation:
- Every function has a docstring (Google style preferred)
- Complex logic includes inline comments explaining "why" not just "what"
- Notebooks include markdown explanations, not just code dumps
- README stays updated as features are added

### Testing:
- Unit tests for data processing functions (pytest)
- Visual validation for CV outputs (save sample images with detections)
- Integration test: Full pipeline on sample video

### Git Practices:
- Frequent commits with clear messages
- Feature branches for major additions
- Main branch always works (CI/CD not required but nice to have)

## Potential Challenges & Solutions

### Challenge: Limited labeled anomaly data
**Solution**: 
- Use reconstruction-based methods (autoencoders)
- Create synthetic anomalies for validation
- Focus on unsupervised metrics (anomaly score distributions)

### Challenge: Underwater image quality varies widely
**Solution**:
- Robust preprocessing pipeline with multiple enhancement techniques
- Test model performance across quality tiers
- Document when model fails and why

### Challenge: Computational constraints
**Solution**:
- Optimize detection model (use lightweight YOLO variants)
- Batch processing for offline analysis
- Downscale video resolution if needed (640x480 vs. 1080p)

### Challenge: Defining "anomaly" is subjective
**Solution**:
- Tunable threshold for anomaly scores
- Provide feature explanations so domain experts can judge relevance
- Separate "unusual" from "concerning" (two-tier system)

## Connections to Target Roles

When building this project, keep these ML Engineer competencies in mind:

### Technical Skills:
- ✅ Computer vision (detection, tracking, image processing)
- ✅ Feature engineering from multi-modal data
- ✅ Anomaly detection (classical + deep learning)
- ✅ Model evaluation and validation
- ✅ Python production code (not just notebooks)

### Portfolio Value:
- Demonstrates end-to-end thinking (data → model → deployment)
- Shows domain adaptation (research methods → industry problem)
- Highlights systems design (scalability, robustness)
- Proves ability to work with messy real-world data

### Resume Bullets This Enables:
- "Built production-ready anomaly detection pipeline processing underwater video at X FPS"
- "Engineered N behavioral and visual features for fish health monitoring using computer vision"
- "Deployed interactive dashboard for real-time anomaly visualization using Streamlit"
- "Achieved X% detection rate on synthetic anomalies with <Y% false positive rate"

## Questions to Consider During Development

These aren't blockers, but thinking points:

1. **Scalability**: If this ran on 100 cameras 24/7, what breaks first?
2. **Explainability**: Can a fish farmer understand WHY the system flagged a fish?
3. **Failure modes**: What causes false alarms? Missed detections?
4. **Generalization**: Does it work on species not in training data?
5. **Maintenance**: How would model performance be monitored in production?

## Communication Style with Ray

✅ **Do**:
- Explain CV concepts clearly (he's learning this domain)
- Provide rationale for technical decisions
- Suggest optimizations with trade-off analysis
- Ask clarifying questions when requirements are ambiguous
- Point out connections to his neuroscience research background

❌ **Don't**:
- Assume prior CV knowledge (but don't over-explain basic Python)
- Give vague advice like "explore different models" without specifics
- Skimp on code comments for complex CV operations
- Ignore deployment/usability considerations

## Success Metrics for This Project

**Technical**:
- [ ] Processes at least 5 minutes of continuous footage
- [ ] Detects >80% of synthetic anomalies
- [ ] False positive rate <10%
- [ ] Explainable feature importance

**Portfolio**:
- [ ] Clean, documented GitHub repo
- [ ] Working Streamlit demo
- [ ] Professional README with results
- [ ] 1-2 blog-worthy insights (e.g., "behavioral features outperform visual features")

**Learning**:
- [ ] Ray feels confident explaining the CV pipeline in interviews
- [ ] Can articulate model trade-offs (precision vs. recall, speed vs. accuracy)
- [ ] Understands deployment considerations for CV systems

---

**Last Updated**: [Today's Date]  
**Primary Contact**: raymond.don.murray@gmail.com  
**Project Repository**: [GitHub URL when created]