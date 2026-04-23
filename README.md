# deep-hedging

Pipeline simple de deep hedging (DDPG) avec sauvegarde automatique de chaque run.

L'objectif est de pouvoir:

- lancer rapidement des expériences,
- conserver les résultats sans relancer,
- réutiliser directement les données pour tableaux/graphes.

## Structure du code

- `main.py`: point d'entrée unique (modes de run, cProfile, persistance).
- `src/orchestrator.py`: logique d'exécution train/eval/benchmark.
- `src/hedging_result.py`: transformation des épisodes en tables exploitables.
- `src/persistence/run_store.py`: écriture des artefacts dans `outputs/`.
- `notebooks/runs_dashboard.ipynb`: lecture et comparaison des runs.

## Exécution rapide

Run complet (train + eval agent + eval benchmark):

```bash
python main.py
```

Run ciblé en CLI:

```bash
python main.py --mode train
python main.py --mode eval_agent
python main.py --mode eval_benchmark
python main.py --mode smoke
```

Choix modulaire du process, de l'agent et du benchmark via CLI:

```bash
python main.py --mode smoke --process GBM --agent DeepDPG --benchmark BsDelta
python main.py --mode smoke --process GBM --agent SkewDPG --benchmark BsDelta
python main.py --mode full --process SABR --agent DeepDPG --benchmark BartlettDelta
```

Piloter facilement la grille temporelle (maturite + rebalancing):

```bash
python main.py --mode full --maturity 1.0 --rebalancing daily
python main.py --mode full --maturity 1.0 --rebalancing weekly
python main.py --mode full --maturity 0.25 --rebalancing daily
python main.py --mode full --maturity 1.0 --n-steps 252
```

Mapping rebalancing -> `n_steps` (par defaut avec `252` jours de bourse/an):

- `daily`, `2d`, `3d`, `weekly`, `biweekly`, `monthly`

Tu peux aussi changer le mapping avec `--trading-days-per-year`.

Run reproductible avec une seed fixe:

```bash
python main.py --mode full --seed 42
```

Tu peux aussi fixer `run.seed` dans `config.json`.

Valeurs disponibles:

- `process`: `GBM`, `SABR`, `SVJ`
- `agent`: `DeepDPG`, `SkewDPG`
- `benchmark`: `BsDelta`, `SABRPractitionerDelta`, `BartlettDelta`

Tu peux aussi piloter le mode dans `config.json` via `run.mode`.

## Modes disponibles

- `full`: train + eval agent + eval benchmark.
- `train`: entraînement uniquement.
- `eval_agent`: évaluation de l'agent uniquement.
- `eval_benchmark`: benchmark uniquement.
- `smoke`: run court pour validation rapide.

## Configuration importante

Le fichier `config.json` contient:

- `simulation`: paramètres de marché (pas, maturité, modèle).
- `training_schedule`: nombre d'épisodes train/eval.
- `hedging_agent`: hyperparamètres DDPG (`actor_learning_rate`, `critic_learning_rate`, `learning_batch_size`, etc.).
  - Pour `SkewDPG`: `skew_lambda`, `skew_penalty`, `skew_eps`.
  - Gradient clipping: `grad_clip` (global) et `grad_clip_q3` (critic Q3).
- `run`: paramètres d'exécution (`mode`, `save_figures`, `enable_cprofile`, profils smoke).

## Outputs produits à chaque run

Chaque exécution crée `outputs/<run_id>/`:

- `config.json`, `meta.json`: reproductibilité du run.
- `data/*_steps.csv`: données pas-à-pas (actions, coûts, rewards).
- `tables/*_episodes.csv`: agrégation par épisode.
- `tables/*_summary.csv`: métriques synthétiques (`mean_total_cost`, `std_total_cost`, `y_objective`).
- `figures/*`: graphes prêts à être utilisés.
- `profile/cprofile.txt` et `profile/cprofile.prof`: profilage CPU.

Le fichier `outputs/runs_index.csv` contient l'historique de tous les runs.

## Notebook de visualisation

Le notebook `notebooks/runs_dashboard.ipynb` permet:

- la liste des runs disponibles,
- l'inspection complète d'un run,
- l'affichage des figures,
- la comparaison inter-runs via `eval_agent_summary.csv`.

Ouvrir le notebook:

```bash
jupyter notebook notebooks/runs_dashboard.ipynb
```

## Optimisation temps de run

- La simulation de paths est lazy (uniquement ce qui est nécessaire au mode choisi).
- Le mode `smoke` évite les runs longs pendant le debug.
- L'index des runs est écrit en append (plus rapide quand l'historique grandit).
- Le profilage peut être désactivé via `run.enable_cprofile` si tu veux aller plus vite.

