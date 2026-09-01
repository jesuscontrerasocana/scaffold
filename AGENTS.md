# AGENTS.md

## Purpose

This repository is a short technical assignment.

Keep the implementation simple, clean, and reliable. The code does not need to support future products, sites, or use cases.

Optimize for:

1. Correctness
2. Simplicity
3. Readability
4. Reliability
5. Easy review and explanation

---

## Coding style

Prefer straightforward Python and small, focused functions.

Keep modules reasonably separated by responsibility, but avoid unnecessary architecture.

Avoid:

* unnecessary abstractions
* base classes
* factories
* registries
* dependency injection
* generic frameworks
* premature generalization
* large refactors unless explicitly requested

Do not create abstractions for hypothetical future requirements.

Prefer explicit code over clever code.

---

## Scope of coding tasks

When given a coding task, implement the requested change with the smallest reasonable modification.

Do not independently redesign the forecasting or optimization methodology.

Do not introduce substantial modeling assumptions unless they are explicitly requested.

If a modeling choice is required to complete a task and the intended behavior is unclear:

* keep the choice simple
* make the assumption visible
* avoid locking the repository into a complex approach

Modeling decisions are discussed separately and recorded in `MODELING_NOTES.md`.

---

## Existing scaffold

Preserve the interfaces and behavior expected by the assignment scaffold.

Do not modify scaffold-owned interfaces unless explicitly requested.

Additional helper modules and tests are fine when they make the implementation clearer.

---

## Data handling

Data handling should be explicit and conservative.

Prefer validation and minimal cleaning over aggressive preprocessing.

Do not silently:

* remove unusual observations
* clip values
* interpolate long gaps
* replace missing values with arbitrary defaults

Make important data-quality handling visible in the code.

---

## Testing

Add focused tests for meaningful behavior when implementing new functionality.

Prefer a few useful tests over extensive testing infrastructure.

Tests should be simple and readable.

---

## Dependencies

Keep dependencies minimal.

Use established libraries where they materially simplify the implementation.

Do not add a dependency for something that can be implemented clearly in a small amount of code.

---

## Runtime and reproducibility

The final pipeline must run reliably using the assignment commands.

Avoid expensive work inside frequently called code unless necessary.

Use deterministic behavior where practical.

Do not depend on local machine state that is not part of the repository.

---

## Comments and documentation

Comments should explain reasoning or non-obvious behavior, not restate the code.

Keep documentation concise.

Do not add large architectural documents unless explicitly requested.

---

## General rule

When multiple implementations are reasonable, prefer the one that is easier to:

* understand
* test
* modify
* explain during the interview

If a requested change can be implemented cleanly without adding a new abstraction, do that.
