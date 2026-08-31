# Data Scientist — technical exercise

## What this is

A small version of something we actually run in production. One commercial site in Belgium:
quarter-hourly metering, rooftop PV, a grid connection with a capacity charge, and a
battery. Forecast what the site will do, decide what the battery should do about it, and
tell us whether it worked.

We have built the plumbing for you. The starter repo handles data loading, the rolling
decision loop, the cost accounting, the plots and the command line — and it ships with a
working baseline forecaster and optimizer, so it runs the moment you unzip it. **You replace
those two files.** That is where we want your time to go.

**Time**: eight hours is a cap, not a target. If you spend less, say so — we would rather
see good judgment about what to leave out than a rushed attempt at everything. If you go
over, tell us where it went.

## Getting started

`scaffold/README.md` walks you through it: install, run the baseline over a month, look at
the plot, then start replacing things. Do that first — it takes five minutes and the picture
tells you more than this document can.

## Part A — Forecast (about 30% of your time)

Forecast what the site will do over the decision horizon, at quarter-hourly resolution,
using only what is known at the moment the decision is made. The harness enforces that for
you; you do not need to engineer around leakage.

`scaffold/README.md` section 3 explains the site from scratch — the power flows and sign
conventions, the battery, the three parts of the bill, and when the prices become known. Read
it even if you know the domain; the day-ahead rule in particular shapes the problem.

Your model is trained once, persisted, and reloaded. Training may be slow. Inference may
not.

## Part B — Formulate and solve (about 50% of your time)

Decide what the battery does over the horizon with the site's energy bill as the objective.

Everything you need is in `site.yaml`. Read all of it — including the parts that are not
about the battery, because they change what "the bill" even means.

You do not need to police the physics: the harness clips whatever the battery cannot
deliver — including any charge or discharge that would drive the meter past the grid
connection limits — and tells you it had to. However, a good formulation would take
those limits into account.

## Part C — Evaluate and report (about 20% of your time)

We are deliberately **not** telling you which metrics to report. Deciding what to measure —
for us, and for the client who is paying for this — is part of the exercise.

The scaffold hands you the bill your schedule produced, computed the same way for every
candidate. It does not tell you whether that bill is good.

## The interview (60 minutes)

The interview has two parts.

**Part 1 — live run (≈ 15 minutes).** At the start of the session we hand you a holdout CSV
covering **1 July to 1 September inclusive** — the same shape as `history.csv`, which only
goes to 31 July. We run your pipeline on it immediately:

```bash
uv run python -m pipeline simulate --data holdout.csv --model-dir models/ \
    --from 2026-08-01 --to 2026-08-31 --out out/august.csv
```

Only August is scored. The July overlap is there so your forecaster has the lags it needs
on the first decision day; the rows past 31 August are there so the day-ahead horizon
resolves correctly on the last scored day.

**Part 2 — presentation and questions (≈ 45 minutes).** You walk us through what you built,
what you measured, and how you arrived at your numbers. Then we ask questions.

**Prepare most of your presentation in advance.** Everything except the handful of holdout
numbers is known before you walk in — build the deck beforehand and leave placeholders for
those final figures.

**The single most important thing you can do to prepare**: rehearse the live run against a
month of `history.csv` you did not develop on. June is the closest analogue to what we will
run.

We are not looking for a perfect solution, but we are looking for **good judgment
and the ability to defend it**.

## Ground rules

- **AI coding assistants are encouraged.** We use them daily. If you found limitations in your own use, tell us about them.
- Python. Any libraries.

## What we are looking for

Someone who can go from a business objective to a formulation, from a formulation to
something that solves reliably, and from a result to an honest account of what it is worth.
We care more about the quality of your decisions and your ability to defend them than about
the volume of your submission.
