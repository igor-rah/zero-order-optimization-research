# Zero-Order Optimization Research

Research on zeroth-order (gradient-free) optimization. Each project lives in its own folder under `projects/` and is self-contained.

## Projects

| Project | Topic | Started | Status |
| --- | --- | --- | --- |
| [`markov-orthogonal-noise`](projects/markov-orthogonal-noise/) | admissible deterministic noise that is orthogonal to the randomization on average over a Markov state; rounding in low-precision MeZO | Sep 2026 | in progress |

## Layout of a project

```
projects/<project-name>/
├── README.md      what the project is about, structure, experiments at a glance, requirements
├── notes/         problem statements, derivations, literature notes
├── experiments/   one folder per experiment: code, raw results (JSON), figures, README
└── paper/         LaTeX sources of the paper (kept out of git until approved)
```

Each experiment folder has its own README: purpose, how to run, how to read the data, main result, limitations. Raw results are committed, so every figure can be rebuilt without re-running the experiments.

## Adding a project

1. Copy [`projects/_template/`](projects/_template/) to `projects/<project-name>/` (lowercase, words separated by hyphens).
2. Fill in its `README.md`.
3. Add a row to the **Projects** table above.
