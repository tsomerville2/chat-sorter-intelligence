**BDD WARP COMMAND**
Usage: /project:bddwarp

Execute a BDD-driven development loop with infinite iterations in the current directory.

**NO ARGUMENTS NEEDED** - Always runs infinite iterations in current directory.

**PHASE 1: INITIAL ASSESSMENT**

Verify BDD setup in current directory:
- Check for features/ directory with .feature files
- Verify features/steps/ directory exists
- Confirm behave (or cucumber) is installed
- Read docs/mission.md, ddd.md, and state-diagram.md for context
- Examine pseudocode/ for implementation guidance

**CRITICAL: Understand the Mission**
Read mission.md and identify:
- What is the ONE main purpose of this app?
- What is the critical path to achieve that purpose?
- Is this a web app (needs browser) or CLI app (needs menu)?
- What would make the user say "this works!"?

Run initial BDD test suite:
```bash
behave --format progress3 --no-capture
```

Capture and analyze:
- Which steps are undefined
- Which steps fail
- Overall test structure

**FAIL-FIRST PRINCIPLE:**
Remember: We WANT tests to fail initially. This is TDD:
- No mocks or stubs
- No "pass" statements
- Love exceptions and errors
- Each failure guides implementation

**PHASE 2: IMPLEMENTATION LOOP**

For iteration = 1 to iterations (or infinite):

**Step 1: Run Tests & Capture State**
```bash
behave --format progress3 --no-capture > test_output.txt
```
Parse output to identify:
- Undefined steps needing implementation
- Failing steps needing code
- Passing steps (victory markers)

**Step 2: Generate Step Definitions**
Deploy Sub Agent for undefined steps:
```
TASK: Generate step definitions for BDD tests

CONTEXT:
- Feature file: [content]
- Undefined steps: [list]
- Tech stack: [from bddinit]

REQUIREMENTS:
1. Create step definition files in features/steps/
2. Each step should RAISE NotImplementedError
3. Include proper behave decorators
4. Match step text exactly
5. Add TODO comments for implementation

DELIVERABLE: Step definition files that fail correctly
```

**Step 3: Implement Domain/Model Layer**
Deploy Sub Agent for data layer:
```
TASK: Implement database models and domain logic

CONTEXT:
- Domain model: [from ddd.md]
- Pseudocode: [from pseudocode/]
- Tech stack: [framework specific]

REQUIREMENTS:
1. Create database models/tables
2. Run migrations if needed (Django: migrate, etc)
3. Implement domain logic from DDD
4. Create test data fixtures
5. Verify database connectivity

DELIVERABLE: Working data layer with test data
```

**Step 4: Implement API Layer**
Deploy Sub Agent for API:
```
TASK: Implement API endpoints

CONTEXT:
- Pseudocode: [main_controller.pseudo]
- Routes needed: [from features]
- Models: [from previous step]

REQUIREMENTS:
1. Create all API endpoints
2. Connect to real database
3. Implement business logic
4. Generate API documentation
5. Test each endpoint manually

DELIVERABLE: Working API with documentation
```

**Step 5: Connect Frontend**
Deploy Sub Agent for UI:
```
TASK: Wire frontend to API

CONTEXT:
- Pseudocode: [web_interface.pseudo]
- API endpoints: [from previous step]
- UI framework: [JavaScript/etc]

REQUIREMENTS:
1. Replace any hardcoded data with API calls
2. Implement real fetch/ajax requests
3. Handle loading states
4. Display real data from backend
5. Ensure error handling

DELIVERABLE: Frontend connected to live API
```

**Step 6: Create User Entry Point**
Deploy Sub Agent for user experience:
```
TASK: Create single-file entry point for users

CONTEXT:
- Mission goal: [from mission.md]
- App type: [web or CLI]
- Tech stack: [from setup]

REQUIREMENTS FOR WEB APPS:
1. Create play.py (or start.py) that:
   - Finds available ports automatically
   - Starts backend API server
   - Serves frontend (if separate)
   - Opens browser to the GAME/APP (not API docs!)
   - Shows "Starting [App Name]..." message
2. Handle Ctrl+C gracefully to stop all services

REQUIREMENTS FOR CLI APPS:
1. Create menu.py using Rich/Textual/Blessed that:
   - Shows beautiful welcome screen
   - Provides numbered menu options
   - Has "1. Quick Start" as first option (no params)
   - Includes help and exit options
   - Uses colors and boxes for visual appeal
2. Make the critical path obvious and immediate

DELIVERABLE: One file that starts everything
```

**Step 7: Integration Testing**
Run full stack test:
```bash
# Use the new entry point
python play.py &  # or menu.py for CLI
APP_PID=$!

# Wait for startup
sleep 3

# Run integration tests
behave --tags=@integration

# Kill app
kill $APP_PID
```

**Step 8: Reality Checks**
Deploy Sub Agent for verification:
```
TASK: Perform user-focused reality verification

REQUIREMENTS:
1. Run the entry point file (play.py/menu.py)
2. Verify it opens to the MAIN PURPOSE immediately
3. For web: Ensure browser opens to game/app (NOT api docs)
4. For CLI: Ensure menu is beautiful and clear
5. Test the critical path as a real user would
6. Verify NO technical barriers between user and goal
7. Check that mission.md goal is achievable in <3 clicks/actions

DELIVERABLE: Reality check report proving user success
```

Deploy Screenshot Sub Agent:
```
TASK: Capture screenshots of running application

CONTEXT:
- App type: [web or CLI]
- Entry point: [play.py or menu.py]

REQUIREMENTS:
1. Create screenshots/ directory
2. For web apps:
   - Use selenium/playwright headless browser
   - Capture: landing page, main interaction, success state
3. For CLI apps:
   - Capture terminal output as images using available tools
   - Alternative: Save text output to screenshots/cli_output.txt
4. Name files descriptively: 01_landing.png, 02_game_in_progress.png, etc.
5. Include a screenshots/README.md explaining what each image shows

DELIVERABLE: screenshots/ folder with captured images and documentation
```

**Step 9: Test Data Verification**
Create and test with realistic data:
- Generate test_data.json with realistic examples
- Create unit tests for data transformations
- Verify edge cases in data flow
- Test validation at each layer

**Step 10: Documentation & README**
Create user-friendly README.md:
```markdown
# [App Name]

[One sentence description from mission.md]

## Quick Start

```bash
python play.py
```

That's it! The game/app will start automatically.

## What This Does
[2-3 sentences about the main purpose]

## Requirements
- Python 3.x
- [Any other requirements]
```

Verify documentation:
- README focuses on USER not developer
- First command is the entry point
- No complex setup instructions
- API docs exist but are secondary

**PHASE 3: LOGGING & EVOLUTION**

**Update droid_log.md:**
```markdown
# Droid Log - Iteration [N]

## Patterns Observed
- [Pattern]: [Description]
- [Challenge]: [How resolved]

## Wild Successes
- [Success]: [What worked well]

## Common Issues
- [Issue]: [Root cause and fix]

## Screenshot Status
- Screenshots captured: [Yes/No]
- Location: screenshots/
- Issues encountered: [Any screenshot setup problems]
```

**Update prompt_evolution.md:**
```markdown
# Prompt Evolution - Iteration [N]

## Improvements for Next Time
- Instead of: "[current approach]"
- Try: "[better approach]"

## Effective Patterns
- "[pattern]" works well for [situation]

## Self-Notes
- [Advice for next iteration]
```

**Step 11: Apply Self-Improvements**
Read prompt_evolution.md and immediately apply suggestions:
- Adjust prompts based on learnings
- Modify approach based on patterns
- Implement suggested improvements
- Focus on user experience improvements

**PHASE 4: LOOP CONTROL**

After each iteration:
1. Check if all tests are passing
2. If yes and iterations != infinite: Complete
3. If no and iterations remaining: Continue
4. If infinite: Continue until context limits

**Progress Tracking:**
```
Iteration [N] of [Total]
- Tests Passing: X/Y
- Coverage: Z%
- Integration: [Status]
```

**PHASE 5: FINAL VERIFICATION**

When all tests pass:
1. Run full test suite one final time
2. Start application and verify manually
3. Document any remaining issues
4. Create summary report

**SUCCESS CRITERIA:**
- All behave tests passing
- Single entry point (play.py/menu.py) works perfectly
- User reaches main goal in <3 actions
- For web: Browser opens to game/app directly
- For CLI: Beautiful menu with quick start
- No technical barriers for users
- README has one simple command to start
- Mission.md goal is immediately achievable

**EXECUTION PRINCIPLES:**

1. **Test-Driven**: Let failing tests drive development
2. **Real Implementation**: No mocks, actual code
3. **Full Stack**: Verify every layer
4. **Continuous Feedback**: Behave is the truth
5. **Self-Improving**: Learn and adapt each iteration
6. **Reality-Based**: Screenshots and manual verification
7. **Data-Centric**: Test actual data flow
8. **Documentation-Aware**: Keep all docs in sync
9. **USER-FIRST**: Always think "How does the user start this?"
10. **CRITICAL PATH**: Focus on mission.md goal above all else

Begin with initial test run and proceed through iterations until success!