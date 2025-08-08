**SSA-LAB: SINGLE SENTENCE ANALYSIS - CREATIVE LABORATORY**

Think deeply about this creative laboratory generation task. You are about to embark on a sophisticated iterative creation process with rigorous testing and fixing.

**Variables:**

spec_file: $ARGUMENTS
output_dir: LABS (always hardcoded)
count: 25 (always hardcoded)

**ARGUMENTS PARSING:**
Parse the following arguments from "$ARGUMENTS":
1. `spec_file` - Path to the markdown specification file OR just a short blurb about the idea
2. `output_dir` (always 'LABS') - Directory where iterations will be saved  
3. `count` - (always 25) Number of iterations to generate

**PHASE 1: SPECIFICATION ANALYSIS**
Read and deeply understand the specification file at `spec_file` or short idea. This file defines:
- What type of content to generate
- The format and structure requirements
- Any specific parameters or constraints
- The intended evolution pattern between iterations
If it's just a short idea, then use best practices that are KISS (keep it simple stupid) because the direct, simple approach is always better. AND, write the spec file in md for reference as we iterate.

Think carefully about the spec's intent and how each iteration should build upon previous work.

**PHASE 2: OUTPUT DIRECTORY RECONNAISSANCE** 
Thoroughly analyze the `LABS` directory to understand the current state:
- List all existing files and their naming patterns
- Identify the highest iteration number currently present
- Analyze the content evolution across existing iterations
- Understand the trajectory of previous generations
- Determine what gaps or opportunities exist for new iterations

**PHASE 3: ITERATION STRATEGY**
Based on the spec analysis and existing iterations:
- Determine the starting iteration number (highest existing + 1)
- Plan how each new iteration will be unique and evolutionary
- Consider how to build upon previous iterations while maintaining novelty
- Prepare for exactly 25 iterations with progressive testing and fixing

**PHASE 4: PARALLEL AGENT COORDINATION WITH TESTING**
Deploy multiple Sub Agents to generate iterations in parallel for maximum efficiency and creative diversity, with enhanced testing protocols:

**Sub-Agent Distribution Strategy:**
- For 25 iterations: Launch in waves of 5 agents to manage coordination and testing
- Each wave completes full testing before next wave begins

**Agent Assignment Protocol:**
Each Sub Agent receives:
1. **Spec Context**: Complete specification file analysis
2. **Directory Snapshot**: Current state of LABS at launch time
3. **Iteration Assignment**: Specific iteration number (starting_number + agent_index)
4. **Uniqueness Directive**: Explicit instruction to avoid duplicating concepts from existing iterations
5. **Quality Standards**: Detailed requirements from the specification
6. **Testing Protocol**: Comprehensive testing and fixing requirements

**Enhanced Agent Task Specification:**
```
TASK: Generate iteration [NUMBER] for [SPEC_FILE] in LABS with comprehensive testing

You are Sub Agent [X] generating iteration [NUMBER]. 

CONTEXT:
- Specification: [Full spec analysis]
- Existing iterations: [Summary of current LABS contents]
- Your iteration number: [NUMBER]
- Assigned creative direction: [Specific innovation dimension to explore]

REQUIREMENTS:
1. Read and understand the specification completely
2. Analyze existing iterations to ensure your output is unique
3. Generate content following the spec format exactly
4. Focus on [assigned innovation dimension] while maintaining spec compliance

TESTING PROTOCOL:
5. **Initial Execution Test**: Try to run/compile the generated content
   - If it's code: Execute it
   - If it's HTML/JS: Test in browser context
   - If it's config/data: Validate format
   - Document execution results

6. **Unit Test Creation**: Write comprehensive unit tests
   - Create test file: iteration_[NUMBER]_test.[ext]
   - Cover core functionality
   - Include edge cases
   - Test innovation features specifically

7. **Iterative Fixing Process**: Fix and test up to 6 times
   - Attempt 1: Run tests, identify failures
   - Attempt 2-6: Fix issues, re-run tests
   - Document each attempt in test_log_[NUMBER].md
   - If still failing after 6 attempts, mark as "experimental" and document known issues

8. **Success Validation**: Ensure iteration passes all tests
   - All unit tests pass
   - Core functionality works
   - Innovation features function correctly
   - No critical errors

9. **Documentation**: Create iteration_[NUMBER]_results.md with:
   - Execution results
   - Test results
   - Fix attempts summary
   - Innovation highlights
   - Known limitations

DELIVERABLE: 
- Main iteration file as specified
- Unit test file
- Test log documentation
- Results summary
```

**Parallel Execution Management with Testing Coordination:**
- Launch waves of 5 Sub Agents simultaneously using Task tool
- Each agent completes full testing cycle before wave completes
- Monitor agent progress through testing phases
- Handle any agent failures by reassigning iteration numbers
- Ensure no duplicate iteration numbers are generated
- Collect and validate all completed iterations with test results
- Move to next wave only after current wave fully tested

**PHASE 5: WAVE-BASED GENERATION WITH QUALITY GATES**
For the 25 iterations, orchestrate 5 waves of parallel generation:

**Wave-Based Generation:**
1. **Wave Planning**: 5 waves of 5 agents each (25 total)
2. **Agent Preparation**: Prepare fresh context snapshots for each wave
3. **Progressive Sophistication**: Each wave explores more advanced innovation dimensions
4. **Quality Gate**: Each wave must complete testing before next wave begins
5. **Learning Integration**: Later waves learn from earlier wave's test results

**Wave Execution Cycle:**
```
FOR wave in [1, 2, 3, 4, 5]:
    1. Assess current LABS state and previous test results
    2. Plan creative directions for 5 agents
    3. Assign increasingly sophisticated innovation goals
    4. Launch 5 parallel Sub Agents
    5. Monitor generation and testing progress
    6. Collect all test results and fixes
    7. Analyze wave outcomes for insights
    8. Update directory state snapshot
    9. Prepare learnings for next wave
```

**Progressive Sophistication Strategy with Testing Focus:**
- **Wave 1 (1-5)**: Basic functional implementations with fundamental testing
- **Wave 2 (6-10)**: Multi-dimensional innovations with interaction testing  
- **Wave 3 (11-15)**: Complex paradigm combinations with integration testing
- **Wave 4 (16-20)**: Advanced features with stress testing and edge cases
- **Wave 5 (21-25)**: Revolutionary concepts with comprehensive validation

**Testing Evolution Across Waves:**
- Early waves: Focus on basic functionality and compilation
- Middle waves: Add interaction and integration testing
- Later waves: Include performance, stress, and edge case testing
- Final wave: Comprehensive test suites with documentation

**EXECUTION PRINCIPLES:**

**Quality & Uniqueness:**
- Each iteration must be genuinely unique and valuable
- Build upon previous work while introducing novel elements
- Maintain consistency with the original specification
- Ensure proper file organization and naming
- **Every iteration must be tested and fixed before completion**

**Testing Rigor:**
- No iteration is complete without passing tests
- Up to 6 fix attempts ensures quality
- Document all testing attempts for learning
- Failed fixes become learning opportunities
- Test results inform future iterations

**Parallel Coordination:**
- Deploy Sub Agents strategically to maximize creative diversity
- Assign distinct innovation dimensions to each agent to avoid overlap
- Coordinate testing phases to prevent resource conflicts
- Monitor all agents for successful completion and quality
- Share test insights across parallel agents

**Scalability & Efficiency:**
- Think deeply about the evolution trajectory across parallel streams
- Balance parallel generation with testing thoroughness
- Use wave-based generation to manage quality gates
- Learn from test failures to improve subsequent waves

**Agent Management:**
- Provide each Sub Agent with complete context and clear assignments
- Include testing protocols in agent instructions
- Handle agent failures gracefully with iteration reassignment
- Ensure all parallel outputs integrate cohesively with test results

**ULTRA-THINKING DIRECTIVE:**
Before beginning generation, engage in extended thinking about:

**Specification & Evolution:**
- The deeper implications of the specification
- How to create meaningful progression across 25 iterations
- What makes each iteration valuable and unique
- How to balance consistency with innovation
- **How testing requirements shape creative decisions**

**Testing Strategy:**
- What constitutes "working" for this specification
- How to write meaningful tests for creative outputs
- Balancing test coverage with innovation freedom
- Learning from test failures to guide evolution
- Creating test suites that validate innovation

**Parallel Strategy:**
- Optimal Sub Agent distribution across 5 waves
- How to assign distinct creative directions to maximize diversity
- Coordinating testing phases across parallel agents
- Managing shared resources during parallel testing
- Quality gate enforcement between waves

**Coordination Challenges:**
- How to prevent duplicate concepts across parallel streams
- Strategies for ensuring each agent produces genuinely unique output
- Managing file naming and directory organization with concurrent writes
- Quality control mechanisms for parallel outputs
- **Coordinating test execution across multiple agents**

**Learning & Adaptation:**
- How test results from early waves inform later waves
- Identifying patterns in successful vs failed iterations
- Adapting creative strategies based on test outcomes
- Building a knowledge base from fix attempts
- Ensuring continuous improvement across all 25 iterations

**Risk Mitigation:**
- Handling agent failures and iteration reassignment
- Ensuring coherent overall progression despite parallel execution
- Managing testing resources across parallel agents
- Maintaining specification compliance across all outputs
- **Gracefully handling iterations that can't be fixed after 6 attempts**

Begin execution with deep analysis of these parallel coordination challenges and testing requirements. Proceed systematically through each phase, leveraging Sub Agents for maximum creative output and ensuring every iteration is tested, fixed, and validated before moving forward. The goal is 25 working, tested, creative iterations that push boundaries while maintaining quality.