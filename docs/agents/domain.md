# Domain Docs

This repo uses a single-context domain documentation layout.

## Before exploring, read these

- `CONTEXT.md` at the repo root, if present
- Relevant ADRs under `docs/adr/`, if present

If these files do not exist, proceed silently. Do not require creating them before work.

## Use the glossary's vocabulary

When output names a domain concept, use the vocabulary from `CONTEXT.md` when available. If the concept you need is not in the glossary yet, note it as a possible gap instead of inventing new project language.

## Flag ADR conflicts

If proposed output contradicts an existing ADR, surface that conflict explicitly rather than silently overriding the decision.
