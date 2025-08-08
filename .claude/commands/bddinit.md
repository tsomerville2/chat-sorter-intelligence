**BDD INITIALIZATION COMMAND**
Usage: /project:bddinit "your app goal"

Initialize a BDD project in the current directory with domain models, state diagrams, and feature files.

**ARGUMENT:**
Parse the app goal from "$ARGUMENTS" - a single string describing what to build.

**TECH STACK DETECTION:**
Analyze the app_goal to detect the appropriate tech stack:
- If mentions "fastapi", "flask", "django" → use python stack
- If mentions "express", "node", "react" → use node stack  
- If mentions "rails", "ruby" → use ruby stack
- Default: python-fastapi

**PHASE 1: ENVIRONMENT SETUP**

Work in the current directory and create:
- features/ directory
- features/steps/ directory  
- docs/ directory for documentation
- pseudocode/ directory for architecture planning

Based on tech_stack, set up the appropriate BDD framework:

**Python Stack:**
- Check if behave is installed: `pip show behave`
- For Django projects: Install behave-django
- For Flask/FastAPI: Install behave
- Create features/environment.py with basic configuration

**Node.js Stack:**
- Check if cucumber is installed: `npm list cucumber`
- Install @cucumber/cucumber if needed
- Create cucumber.js configuration file

**Ruby Stack:**
- Check if cucumber is installed: `gem list cucumber`
- Install cucumber gem if needed
- Create cucumber.yml configuration

**PHASE 2: GOAL ANALYSIS & APP NAMING**

Analyze the app_goal to extract:
- Core domain entities
- Primary user actions
- Key business processes
- Success criteria

Generate an appropriate app name:
- Extract key concepts from the goal
- Create a memorable, descriptive name
- Ensure it's suitable for the domain

**PHASE 3: DOMAIN-DRIVEN DESIGN DOCUMENT**

Create `docs/ddd.md` with minimal, essential domain model:

```markdown
# Domain Model - [App Name]

## Bounded Context
[Single bounded context for this simple app]

## Aggregates
[List only essential aggregates - usually 1-3 for simple apps]

### [Aggregate Name]
- **Root Entity**: [Entity name]
- **Value Objects**: [List if any, keep minimal]
- **Business Rules**: [Core invariants only]

## Domain Events
[List 2-4 critical events that drive the system]

## Ubiquitous Language
[5-10 key terms max, with clear definitions]
```

Focus on:
- Only entities that directly serve the app goal
- Remove any "nice-to-have" concepts
- Keep relationships simple
- No technical implementation details

**PHASE 4: STATE DIAGRAM GENERATION**

Create `docs/state-diagram.md` with Mermaid stateDiagram:

```markdown
# State Flow - [App Name]

## Business State Diagram

\```mermaid
stateDiagram-v2
    [*] --> Initial
    Initial --> [Core State 1]
    [Core State 1] --> [Core State 2]: [Action]
    [Core State 2] --> [End State]: [Completion]
    [End State] --> [*]
\```

## State Definitions
- **Initial**: [What triggers the process]
- **[Core States]**: [What happens in each state]
- **[End State]**: [Success condition]

## Transitions
[List each transition with business rules]
```

Rules for state diagram:
- Maximum 5-7 states total
- Only happy path transitions
- No error states unless critical
- Clear start and end

**PHASE 5: MISSION DOCUMENT**

Create `docs/mission.md`:

```markdown
# Mission - [App Name]

## Vision
[Elaborate the one-sentence goal into 2-3 paragraphs]

## Success Criteria
1. [Specific, measurable outcome 1]
2. [Specific, measurable outcome 2]
3. [Specific, measurable outcome 3]

## In Scope
- [Core feature 1]
- [Core feature 2]
- [Core feature 3]

## Out of Scope
- [Explicitly excluded feature 1]
- [Explicitly excluded feature 2]
- [Future enhancement 1]

## App Name Rationale
**Chosen Name**: [App Name]
**Reasoning**: [Why this name fits the mission]
```

**PHASE 6: MINIMAL FEATURE FILES**

Create ultra-minimal feature files - first pass:

1. Identify all possible features
2. Reduce to happy paths only
3. Further reduce to critical path only

Final features should be:
- One core workflow feature
- Optional: One setup feature (only if required)
- Maximum 3-5 scenarios total across all features

Example `features/core_workflow.feature`:
```gherkin
Feature: [Core Workflow Name]
  As a [primary user]
  I want to [primary action]
  So that [primary value]

  Scenario: [Single Critical Path]
    Given [minimal precondition]
    When [essential action]
    Then [core success criteria]
```

**PHASE 7: 1990s PSEUDOCODE ARCHITECTURE**

Generate strict procedural pseudocode with:
- Clear BEGIN/END blocks
- Explicit variable declarations
- Simple procedural flow
- No patterns or abstractions

Structure:
```
pseudocode/
├── main_controller.pseudo
├── data_manager.pseudo
├── business_rules.pseudo
└── io_handler.pseudo
```

Example format:
```
PROGRAM MainController
BEGIN
    DECLARE userInput AS STRING
    DECLARE dataStore AS DataManager
    DECLARE result AS BOOLEAN
    
    FUNCTION ProcessRequest(input)
    BEGIN
        VALIDATE input
        IF input IS VALID THEN
            result = dataStore.Save(input)
            RETURN result
        ELSE
            RETURN FALSE
        END IF
    END
    
    // Main execution
    userInput = GetUserInput()
    result = ProcessRequest(userInput)
    DisplayResult(result)
END
```

**PHASE 8: ARCHITECTURE REVIEW & SIMPLIFICATION**

Review all pseudocode and ask:
1. Can any two modules be combined?
2. Is there any unnecessary indirection?
3. Could this be done with fewer files?
4. Would a beginner understand this immediately?

Simplify until the answer to #4 is absolutely YES.

**PHASE 9: SUMMARY GENERATION**

Create `summary.md`:

```markdown
# BDD Project Initialized - [App Name]

## Generated Structure
- ✅ BDD framework configured ([behave/cucumber])
- ✅ Domain model defined (docs/ddd.md)
- ✅ State flow mapped (docs/state-diagram.md)
- ✅ Mission clarified (docs/mission.md)
- ✅ Features created ([list feature files])
- ✅ Architecture planned ([list pseudocode files])

## Quick Start
1. Review the generated documents in docs/
2. Examine the features/ directory
3. Check pseudocode/ for the planned architecture

## Next Steps
Run the bddloop command to:
- Generate step definitions
- Implement the pseudocode as real code
- Make all tests pass

## Configuration
- Tech Stack: [chosen stack]
- BDD Framework: [behave/cucumber]
- App Goal: "[original goal]"
```

**EXECUTION PRINCIPLES:**

1. **Ruthless Simplification**: Always choose the simpler option
2. **No Gold Plating**: Only what directly serves the stated goal
3. **Clear Over Clever**: 1990s clarity beats modern patterns
4. **Test-First Thinking**: Everything prepared for BDD implementation
5. **Single Responsibility**: Each component does ONE thing

Begin by parsing arguments and systematically work through each phase, constantly asking "Can this be simpler?"