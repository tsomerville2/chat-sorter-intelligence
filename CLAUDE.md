# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental project demonstrating the SSA-LAB (Single Sentence Analysis - Laboratory) pattern using Claude Code's custom slash commands. The project orchestrates multiple AI agents in parallel to generate, test, and fix evolving iterations of creative implementations based on specifications or simple ideas.

## Key Commands

### Running the SSA-LAB Creative Laboratory

```bash
claude
```

Then use the `/project:ssa-lab` slash command with these variants:

```bash
# Basic usage (always generates 25 tested iterations in LABS directory)
/project:ssa-lab .claude/example_specs/specs_html_javascript_ui/invent_new_ui_v3.md

# Short idea usage
/project:ssa-lab "Create innovative React component patterns"

# Different spec file
/project:ssa-lab .claude/example_specs/specs_python_pyqt6_ui/create_pyqt6_app_v1.md

# Note: Always generates 25 iterations with testing in the LABS directory
```

## Architecture & Structure

### Command System
The project uses Claude Code's custom commands feature:
- `.claude/commands/ssa-lab.md` - SSA-LAB creative laboratory command with testing
- `.claude/commands/prime.md` - Additional command (if present)
- `.claude/settings.json` - Permissions configuration allowing Write, MultiEdit, Edit, and Bash

### Additional Resources
- `.claude/ai_docs/claude_code_fresh_tutorials.md` - Comprehensive Claude Code tutorials and documentation

### Specification-Driven Generation
- Specifications in `.claude/example_specs/specs_html_javascript_ui/` directory define what type of content to generate
- Available specs: `invent_new_ui_v2.md`, `invent_new_ui_v3.md`, `invent_new_ui_v4.md` - UI Component Specifications
- Specs define naming patterns, content structure, design dimensions, and quality standards

### Multi-Agent Orchestration Pattern
The ssa-lab command implements sophisticated parallel agent coordination with testing:
1. **Specification Analysis** - Deeply understands the spec requirements
2. **Directory Reconnaissance** - Analyzes existing iterations to maintain uniqueness
3. **Parallel Sub-Agent Deployment** - Launches multiple agents with distinct creative directions
4. **Testing & Fixing Protocol** - Each iteration tested and fixed up to 6 times
5. **Wave-Based Generation** - Manages 5 waves of 5 agents for 25 total iterations
6. **Context Management** - Optimizes context usage across all agents

### Generated Content Organization
- `LABS/` - Primary output directory for all tested iterations
- Each iteration includes main file, tests, test logs, and results documentation
- Failed iterations moved to `LABS/experimental/` with documentation

### Key Implementation Details
- Sub-agents receive complete context including spec, existing iterations, and unique creative assignments
- Parallel execution managed through Task tool with waves of 5 agents
- Progressive sophistication strategy across 5 waves
- Each iteration must be genuinely unique, tested, and working
- Testing includes compilation, unit tests, and iterative fixes