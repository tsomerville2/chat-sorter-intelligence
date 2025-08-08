# SSA-SCAN: Single Sentence Analysis - Technology Scanner & Prover

## Command Usage
```
/project:ssa-scan <sentence describing technology needs or concept>
```

## Examples
```
/project:ssa-scan "I need to build a real-time collaborative whiteboard with WebRTC and Canvas"
/project:ssa-scan "Show me different ways to implement voice-to-text transcription in Python"
/project:ssa-scan "Demonstrate various React state management libraries for large applications"
```

## Execution Flow

### Phase 1: Analysis & Technology Discovery
1. Parse the input sentence to extract key requirements
2. If specific technologies mentioned: extract them
3. If only concept described: perform web search to discover top 10 relevant technologies
4. Query Context7 MCP for additional insights and recommendations
5. Create `/SCAN/scan.md` with:
   - Original request
   - Extracted requirements
   - Technology list with brief descriptions
   - Web search findings (if applicable)
   - Context7 recommendations (if available)

### Phase 2: Parallel Proof-of-Concept Generation
For each identified technology (up to 10), launch parallel agents:

#### Sub-Agent A: Environment Preparation
- Analyze technology requirements
- Generate installation commands (npm, pip, cargo, etc.)
- Create setup scripts
- Document dependencies in `/SCAN/[tech_name]/setup.md`

#### Sub-Agent B: Focused Implementation
- Create minimal but complete proof-of-concept
- File: `/SCAN/[tech_name]/poc.{ext}`
- Include:
  - Core functionality demonstration
  - Key features highlighted
  - Clean, well-commented code
  - README.md with usage instructions

#### Sub-Agent C: Testing & Validation
- Generate comprehensive unit tests
- File: `/SCAN/[tech_name]/test.{ext}`
- Execute tests with up to 6 retry attempts:
  1. Run initial test
  2. If failed: analyze error, fix code
  3. Re-run test
  4. Document each attempt in `/SCAN/[tech_name]/test_log.md`
  5. Continue until success or 6 attempts exhausted

#### Sub-Agent D: Documentation & Screenshots
- If tests pass:
  - Generate visual proof (screenshot/output capture)
  - Save as `/SCAN/[tech_name]/proof.png`
  - Create success summary
- If tests fail after 6 attempts:
  - Move entire folder to `/SCAN/JUNK/[tech_name]_failed/`
  - Document failure reasons

### Phase 3: Results Aggregation
Update `/SCAN/scan.md` with:
- ✅ Successful implementations with links to code
- ❌ Failed attempts with failure analysis
- 📊 Comparison matrix of features/capabilities
- 🏆 Recommendations based on results
- 📈 Performance metrics where applicable

## Directory Structure
```
/SCAN/
├── scan.md                          # Master results document
├── web_search_results.json          # Raw search data
├── context7_insights.md             # MCP recommendations
├── [tech_name_1]/                   # Successful POC
│   ├── setup.md                     # Installation guide
│   ├── poc.{ext}                    # Main implementation
│   ├── test.{ext}                   # Test suite
│   ├── test_log.md                  # Test execution history
│   ├── proof.png                    # Visual proof
│   └── README.md                    # Usage documentation
├── [tech_name_2]/                   # Another successful POC
│   └── ...
└── JUNK/                            # Failed attempts
    └── [tech_name_failed]/
        └── ... (all files moved here)
```

## Implementation Details

### Technology Detection Patterns
```python
# Extract explicit technology mentions
tech_patterns = [
    r'\b(React|Vue|Angular|Svelte)\b',
    r'\b(WebRTC|Socket\.io|WebSockets)\b',
    r'\b(TensorFlow|PyTorch|Keras)\b',
    r'\b(FastAPI|Flask|Django|Express)\b',
    # ... comprehensive pattern list
]

# Concept-to-technology mapping
concept_map = {
    'real-time': ['WebSockets', 'Socket.io', 'WebRTC', 'Firebase'],
    'machine learning': ['TensorFlow', 'PyTorch', 'Scikit-learn'],
    'state management': ['Redux', 'MobX', 'Zustand', 'Recoil'],
    # ... extensive mapping
}
```

### Test Retry Logic
```python
for attempt in range(1, 7):
    result = run_tests()
    if result.success:
        capture_proof()
        break
    else:
        analyze_failure(result.error)
        apply_fix(attempt, result.error)
        document_attempt(attempt, result)
```

### Success Criteria
- Code compiles/runs without errors
- All unit tests pass
- Key functionality demonstrated
- Visual proof captured (where applicable)
- Documentation complete

## Special Features

### 1. Smart Technology Selection
- Prioritizes actively maintained libraries
- Considers ecosystem compatibility
- Balances popularity with innovation

### 2. Adaptive Testing
- Language-specific test runners
- Framework-aware assertions
- Automatic test generation based on code structure

### 3. Visual Proof Generation
- Browser automation for web technologies
- Terminal output capture for CLI tools
- GUI screenshots for desktop applications
- API response formatting for services

### 4. Failure Analysis
- Dependency conflicts detection
- Version incompatibility warnings
- Alternative solution suggestions
- Learning from failures for future runs

## Ultra-Thinking Directive

Before processing each SSA-SCAN request:

**Technology Discovery:**
- What technologies directly address the stated need?
- Which alternatives offer unique advantages?
- What's the current industry standard vs emerging solutions?
- How do these technologies complement each other?

**Implementation Strategy:**
- What's the minimal code needed to prove viability?
- Which features are essential vs nice-to-have?
- How can we ensure fair comparison between options?
- What metrics matter most for this use case?

**Testing Philosophy:**
- What constitutes "proof" for this technology?
- Which edge cases are critical to test?
- How do we simulate real-world usage?
- What performance benchmarks apply?

**Documentation Excellence:**
- What would a developer need to go from POC to production?
- How do we make results immediately actionable?
- What visual proof best demonstrates success?
- Which failure patterns help future developers?

**Continuous Improvement:**
- How do we learn from each scan to improve future ones?
- What patterns emerge across similar technology requests?
- How can we optimize the retry logic for specific tech stacks?
- What additional Context7 queries would enhance results?

## Execution Commitment

The SSA-SCAN command operates with relentless determination:
- **Completes 150% of the task** - goes beyond basic requirements
- **Never stops at first failure** - exhausts all retry attempts
- **Covers every mentioned technology** - no cherry-picking
- **Provides actionable results** - not just pass/fail
- **Learns and adapts** - each run improves the process

## Output Quality Standards

Every scan produces:
- **Working code** or detailed failure analysis
- **Reproducible results** with exact versions
- **Visual proof** of successful execution
- **Comparison insights** between technologies
- **Production-ready starter code** for chosen solution
- **Decision matrix** for technology selection

This command transforms a single sentence into a comprehensive technology evaluation suite, providing concrete proof-of-concepts that demonstrate real capabilities, not just theoretical possibilities.