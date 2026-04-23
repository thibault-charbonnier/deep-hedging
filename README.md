# deep-hedging

Pipeline de **deep hedging** pour la couverture d'options européennes via *Deep Deterministic Policy Gradient* (DDPG), inspiré de **Cao et al. (2021)** "Deep Hedging of Derivatives Using Reinforcement Learning". L'agent apprend une politique de couverture sous coûts de transaction et la compare à des benchmarks analytiques (BS delta, Bartlett delta).

---

## Sommaire

1. [Fonctionnement général](#1-fonctionnement-général)
2. [Installation](#2-installation)
3. [Structure du projet](#3-structure-du-projet)
4. [Lancer un run](#4-lancer-un-run)
5. [Le fichier `config.json`](#5-le-fichier-configjson)
6. [Artefacts produits par un run](#6-artefacts-produits-par-un-run)
7. [Visualisation (`notebooks/plot.ipynb`)](#7-visualisation-notebooksplotipynb)
8. [Notes théoriques](#8-notes-théoriques)

---

## 1. Fonctionnement général

À chaque run, le pipeline exécute trois étapes :

1. **`train`** — l'agent apprend sur `train_episodes` paths simulés.
2. **`eval_agent`** — la policy figée est évaluée sur `eval_episodes` paths.
3. **`eval_benchmark`** — le benchmark analytique (ex. BS delta) est évalué sur les **mêmes** paths d'évaluation, ce qui permet une comparaison directe.

### Boucle d'un épisode

Pour chaque path `(S_0, S_1, …, S_n)` :

- **Setup (t=0)** : l'agent choisit une couverture initiale `H_0` ; un coût `κ·S_0·|H_0|` est facturé.
- **Pas courants (i = 0, …, n−2)** : l'agent choisit `H_{i+1}` ; la reward est
  ```
  R_{i+1} = (V_{i+1} − V_i) + H_i·(S_{i+1} − S_i) − κ·|S_{i+1}·(H_{i+1} − H_i)|
  ```
  où `V_i` est le prix BS de l'option et `κ` le coût de transaction proportionnel.
- **Pas terminal (i = n−1)** : la position est **liquidée de force** (`H_n := 0`). Coût = `κ·S_n·|H_{n−1}|`. La reward de liquidation est **fusionnée** dans la transition précédente (schéma *buffered-commit* — évite que le critique soit appelé sur un état/action hors-distribution au pas terminal, source classique de divergence de la loss).

### Cascade état → action → reward

- **État** (4 dimensions) :
  `[holding H_i, log(S/K), TTM/T, σ_t / σ_ref]`
  où `σ_ref = σ_0` du process simulé (auto-détecté depuis la première valeur du vol-path).
- **Action** : la policy de DDPG renvoie une fraction de couverture dans `[action_low, action_high]`.
- **Reward** : P&L comptable pas-à-pas (Section 3.1 du papier).

---

## 2. Installation

Python ≥ 3.10. Dépendances principales :

```bash
pip install numpy scipy pandas matplotlib torch cpprb rich
```

Pour le notebook de plots :

```bash
pip install jupyter seaborn
```

---

## 3. Structure du projet

```
deep-hedging/
├── main.py                       # Point d'entrée unique
├── config.json                   # Tous les hyperparamètres
├── src/
│   ├── orchestrator.py           # Boucles train / test / test_benchmark
│   ├── hedging_strategy/
│   │   └── hedging_env.py        # Env de hedging (rewards, state)
│   ├── simulation/
│   │   ├── gbm_process.py        # Black-Scholes (GBM)
│   │   ├── sabr_process.py       # SABR β=1
│   │   └── svj_process.py        # Stochastic vol + sauts (Heston-like)
│   ├── valuation/
│   │   ├── bs_valuation.py       # Prix et delta BS vectorisés
│   │   └── sabr_valuation.py     # Hagan implied vol, Bartlett delta
│   ├── benchmark/
│   │   ├── bs_delta.py                  # Delta BS classique (pour GBM)
│   │   ├── sabr_practitioner_delta.py   # BS delta avec σ_impl SABR
│   │   └── bartlett_delta.py            # Bartlett delta (pour SABR)
│   ├── hedging_agents/
│   │   ├── ddpg_agent.py         # DDPG avec critiques Q1/Q2 (mean-std)
│   │   └── qr_ddpg_agent.py      # DDPG distributionnel (quantile regression + CVaR)
│   ├── persistence/
│   │   └── run_store.py          # Écriture des artefacts dans outputs/
│   ├── utils/
│   │   ├── helpers.py            # Seeding, I/O JSON, ensure_dir
│   │   ├── run_plots.py          # Fonctions de plotting
│   │   └── enums.py              # Registres Process/Agent/Benchmark
│   └── hedging_result.py         # Aggrégation épisode → DataFrames
├── notebooks/
│   └── plot.ipynb                # Analyse / comparaison des runs
└── outputs/                      # Un sous-dossier par run (voir §6)
    └── runs_index.csv            # Index global de tous les runs
```

---

## 4. Lancer un run

### Run standard

```bash
python main.py
```

Tout est configuré via `config.json`. La CLI n'expose qu'un argument optionnel :

```bash
python main.py --config chemin/vers/autre_config.json
```

### Ce que fait `main.py`

1. Charge `config.json`.
2. Fixe la seed si `run.seed` est défini (reproductibilité totale : numpy, random, torch).
3. Calcule `n_steps = round(maturity · 252 / rebalancing)` à partir de `run.maturity` (années) et `run.rebalancing` (jours).
4. Lance le pipeline `train` → `eval_agent` → `eval_benchmark`.
5. Sauvegarde tous les artefacts sous `outputs/<run_id>/`.

---

## 5. Le fichier `config.json`

C'est le **seul** endroit où paramétrer un run.

### `simulation` — paramètres du process de marché

```json
"simulation": {
    "maturity": 0.0833333333,
    "S0": 100.0,
    "gbm":  { "mu": 0.05, "sigma": 0.2 },
    "sabr": { "mu": 0.05, "sigma0": 0.2, "nu": 0.6, "rho": -0.4 },
    "svj":  { "mu": 0.05, "v0": 0.04, "kappa": 1.0, "theta": 0.04,
              "xi": 0.5, "rho": -0.5,
              "jump_intensity": 1.0, "jump_mean": -0.02, "jump_std": 0.05 }
}
```

- **`maturity`** : horizon du dérivé (années). `0.0833 ≈ 1 mois`, `0.25 = 3 mois`, `1.0 = 1 an`.
- **`S0`** : spot initial (par convention 100 → option ATM avec `strike=100`).
- **GBM** : `mu` est le drift, `sigma` la vol constante. Pour tester la convergence vers le delta BS, utilise GBM.
- **SABR β=1** : `sigma0` est la vol initiale, `nu` la vol-of-vol (plus grand → skew/smile plus marqué), `rho` la corrélation S/σ (négative → skew négatif typique equities).
- **SVJ** : modèle Heston + sauts poissoniens. `v0, kappa, theta, xi, rho` contrôlent la diffusion de variance ; `jump_intensity, jump_mean, jump_std` paramétrisent les sauts lognormaux. Typique : `jump_mean < 0` pour des sauts de crash.

### `hedging_env` — paramètres d'exécution

```json
"hedging_env": {
    "position_sign": -1.0,
    "transaction_cost": 0.01
}
```

- **`position_sign`** : `-1.0` = on vend l'option et on se couvre ; `+1.0` = on achète. Inverse le signe du prix `V_i` dans la reward.
- **`transaction_cost`** (`κ`) : proportionnel à `S·|ΔH|`. `0.01` = 1%. À **0**, l'optimum est le delta analytique. Plus `κ` est grand, plus l'agent doit lisser ses rebalancements.

### `training_schedule` — budget d'apprentissage

```json
"training_schedule": {
    "train_episodes": 15000,
    "eval_episodes": 3000,
    "update_frequency": 5
}
```

- **`train_episodes`** : nombre de paths d'entraînement. Ordre de grandeur : 10k–50k pour converger proprement sur GBM, davantage sur SABR/SVJ.
- **`eval_episodes`** : nombre de paths d'évaluation. 3000 donne une incertitude raisonnable sur `mean/std` du coût total.
- **`update_frequency`** : on appelle `learn()` tous les N pas. `1` = update à chaque pas (coûteux), `5` est un bon compromis vitesse/qualité.

### `hedging_agent` — hyperparamètres DDPG

```json
"hedging_agent": {
    "exploration_rate_start": 1.0,
    "exploration_rate_end": 0.05,
    "exploration_rate_decay": 0.9999,
    "actor_learning_rate": 0.0001,
    "critic_learning_rate": 0.001,
    "discount_factor": 1.0,
    "learning_batch_size": 128,
    "replay_capacity": 200000,
    "min_buffer_size": 1024,
    "target_update_freq": 100,
    "risk_lambda": 1.5,
    "n_quantiles": 51,
    "cvar_alpha": 0.95,
    "huber_kappa": 1.0,
    "action_low": 0.0,
    "action_high": 1.0
}
```

- **Exploration ε-greedy** : démarre à `exploration_rate_start` puis décroît multiplicativement de `exploration_rate_decay` à chaque pas d'apprentissage, jusqu'à `exploration_rate_end`. Avec `decay = 0.9999`, il faut ≈ 10 000 updates pour passer de 1.0 à 0.37.
- **Learning rates** : convention DDPG — actor plus lent (`1e-4`) que critique (`1e-3`). Le critique doit converger en premier pour fournir un gradient propre à l'actor.
- **`discount_factor` (γ)** : `1.0` est usuel sur ce problème (horizon fini, pas d'actualisation puisque `maturity` est court et `rf = 0`).
- **`learning_batch_size`** : taille du minibatch échantillonné du replay buffer. 64–256 est standard.
- **`replay_capacity`** : taille max du buffer. Doit être ≥ `train_episodes × n_steps`. À 200k avec ~21 pas/épisode × 15k épisodes = 315k, on sature un peu → augmenter si besoin, ou garder tel quel (FIFO est acceptable pour ce problème).
- **`min_buffer_size`** : aucun `learn()` avant que le buffer contienne ≥ N transitions. Évite d'apprendre sur 1–2 samples au début.
- **`target_update_freq`** : hard copy des réseaux target tous les N updates (convention DQN, pas de soft update ici).
- **`risk_lambda` (λ_std)** : trade-off mean-std dans la loss de l'actor (DeepDPG uniquement) : `F = E[C] + λ · std(C)`. `1.5` correspond au papier. Plus grand → couverture plus conservative. Ignoré par QRDDPG qui a son propre objectif (CVaR).
- **Paramètres QR-DDPG (`QRDDPG`)** :
  - **`n_quantiles`** : nombre de quantiles τᵢ = (i − 0.5)/N approximant la distribution du coût. 51 par défaut (standard QR-DQN), plus = distribution plus fine mais plus coûteux.
  - **`cvar_alpha`** : niveau du CVaR minimisé par l'actor (ex. 0.95 = moyenne des 5% pires cas). Unique objectif de l'actor.
  - **`huber_kappa`** : seuil du Huber loss pour la quantile regression (robuste aux outliers, défaut 1.0).
- **`action_low` / `action_high`** : bornes de la holding `H`. `[0, 1]` pour une short call couverte par un long sur le sous-jacent (0 ≤ H ≤ 1).

### `derivative` — paramètres de l'option

```json
"derivative": {
    "strike": 100.0,
    "rf_rate": 0.0,
    "div_rate": 0.0,
    "option_type": "call"
}
```

- **`option_type`** : optionnel, `"call"` par défaut. Peut valoir `"put"`.

### `run` — métadonnées du run

```json
"run": {
    "maturity": 0.0833333333,
    "rebalancing": 1,
    "process": "GBM",
    "agent": "DeepDPG",
    "benchmark": "BsDelta",
    "seed": 42
}
```

- **`maturity`** : dupliqué avec `simulation.maturity` (main.py écrase `simulation.maturity` par la valeur de `run.maturity` pour simplifier les scripts).
- **`process`** : `"GBM"` | `"SABR"` | `"SVJ"`.
- **`agent`** : `"DeepDPG"` | `"QRDDPG"`.
- **`benchmark`** : `"BsDelta"` | `"BartlettDelta"` | `"SABRPractitionerDelta"`.

- **`rebalancing`** : espacement entre rebalancements (jours de bourse). `n_steps = round(maturity · 252 / rebalancing)`. Ex. `maturity=0.25, rebalancing=1` → 63 pas.
- **`process / agent / benchmark`** : doivent matcher les noms exposés dans `src/utils/enums.py`.
- **`seed`** : si fourni, fixe numpy + random + torch.

### Règles de cohérence à connaître

- Si tu changes de `process`, **change aussi le `benchmark`** correspondant :
  - GBM → `BsDelta`
  - SABR → `BartlettDelta` (ou `SABRPractitionerDelta`)
  - SVJ → pas de benchmark dédié ; `BsDelta` est le plus raisonnable (c'est un benchmark "naïf").
- `σ_ref` pour la normalisation de l'état est maintenant pris de `_vol_path[0]`, c'est-à-dire automatiquement `gbm.sigma`, `sabr.sigma0`, ou `√svj.v0` selon le process.
- **Limitation connue** : la valuation de l'option à chaque pas se fait en **BS avec σ constant = `gbm.sigma`**, même sous SABR/SVJ. Sous ces modèles, la reward sous-estime le vega P&L (sujet de correction à faire — voir §8).

---

## 6. Artefacts produits par un run

Chaque run crée `outputs/<YYYYMMDD_HHMMSS>_<process>_<agent>_<benchmark>_<hash6>/`.

```
<run_id>/
├── config.json              # Copie exacte du config utilisé
├── meta.json                # run_id, timestamp, version Python, seed
├── data/
│   ├── train_steps.csv              # 1 ligne par pas × épisode (train)
│   ├── eval_agent_steps.csv         # idem (eval agent)
│   └── eval_benchmark_steps.csv     # idem (eval benchmark)
└── tables/
    ├── *_episodes.csv       # agrégation par épisode (total_cost, loss moyenne…)
    └── *_summary.csv        # résumé : mean_total_cost, std_total_cost, y_objective
```

### Colonnes importantes

**`*_steps.csv`** (pas-à-pas) :
- `split, episode_idx, step_idx, time, time_next, spot, spot_next`
- `action` : holding après décision
- `reward, cost` : reward comptable et son opposé (cost = -reward)
- `trade_cost, liquidation_cost` : décomposition du coût de transaction
- `loss` : valeur de la loss agent au pas (NaN hors updates)
- `sigma, variance` : ajoutées si le process les expose

**`*_summary.csv`** :
- `mean_total_cost, std_total_cost, skew_total_cost`
- `y_objective = mean + risk_lambda × std` (objectif Cao mean-std)

### Rescaling par `option_price_t0`

Tous les coûts sont **normalisés** (multipliés par `100 / option_price_t0`) avant écriture. `option_price_t0` = prix BS de l'option à t=0 avec paramètres de `config`. Les coûts sont donc en "bps du prix initial" → comparables entre maturités/strikes.

### Index global

`outputs/runs_index.csv` enregistre chaque run (run_id, timestamp, ok, note) — utile pour tracker l'historique depuis un notebook.

---

## 7. Visualisation (`notebooks/plot.ipynb`)

Le notebook `notebooks/plot.ipynb` charge un ou plusieurs runs depuis `outputs/` et produit :

- **Distribution de coût total** (histogramme train / eval / benchmark sur le même plot).
- **Trajectoires moyennes** du holding agent vs. benchmark le long d'un path moyen.
- **Scatter plots** (`prev_action` vs. `action`) pour visualiser la stabilité de la policy.
- **Courbes de loss** et d'`epsilon` durant l'entraînement.
- **Comparaison inter-runs** via concatenation des `*_summary.csv`.

Pour lancer :

```bash
jupyter notebook notebooks/plot.ipynb
```

Les fonctions de plotting réutilisables sont dans `src/utils/run_plots.py`.

---

## 8. Notes théoriques

### Convention P&L comptable

Paper Section 3.1 : `R_{i+1} = V_{i+1} − V_i + H_i(S_{i+1} − S_i) − κ·|S_{i+1}(H_{i+1} − H_i)|`

avec coût initial `−κ·|S_0·H_0|` et coût final `−κ·|S_n·H_n|` (= 0 car `H_n := 0` forcé).

### Gestion du pas terminal

`H_n = 0` est forcé dans l'env au pas terminal (liquidation automatique, pas de décision policy). Dans `Orchestrator.train()`, on utilise un **buffered-commit** :

- La transition du pas `i = n−2` n'est **pas** stockée immédiatement.
- Au pas terminal, on **fusionne** sa reward avec la reward de liquidation et on la stocke avec `done=True`.
- La transition terminale isolée `(s_{n−1}, a=0, r_liq, s_n, True)` n'entre **jamais** dans le replay buffer.

**Pourquoi :** sans ce schéma, le critique ne voyait `Q(s_{n−1}, a)` que pour `a = 0`, et son extrapolation hors-distribution était utilisée dans le bootstrap target à `i = n−2`, causant une divergence classique par *extrapolation error* (Fujimoto et al.).

### Agent QR-DDPG (distributionnel)

Au lieu de prédire un scalaire `E[C]`, le critique prédit la **distribution** du coût via N quantiles `θᵢ(s, a) ≈ F⁻¹_{C|s,a}(τᵢ)` avec `τᵢ = (i − 0.5)/N`.

- **Critic loss** : quantile Huber regression (Dabney et al. 2018, QR-DQN).
- **Actor loss** : minimise le **CVaR_α** de la distribution prédite — moyenne des quantiles au-delà du niveau α (`α=0.95` → moyenne des 5% pires coûts).
- **Avantages** : un seul critique, CVaR directement interprétable en finance (Basel, Solvency II), tous les moments dérivables en post-process (mean, std, skew, VaR…).
- **Hyperparams-clés** : `n_quantiles` (51 défaut), `cvar_alpha` (0.95), `huber_kappa` (1.0).

### Benchmarks

- **`BsDelta`** : delta BS classique avec σ = `gbm.sigma`. Optimal sous GBM sans frictions.
- **`SABRPractitionerDelta`** : calcule σ_impl via Hagan puis applique la formule BS delta. Benchmark "naïf" sous SABR.
- **`BartlettDelta`** (Bartlett 2006) : `Δ_BS(σ_impl) + vega_BS · ρν/S`. Corrige pour l'anticipation de `dσ` sachant `dS`. Benchmark optimal sous SABR.

### Limitation courante (non-corrigée)

La valuation de l'option dans `HedgingEnv` utilise toujours `BSValuation` avec `σ = gbm.sigma` constant. Sous SABR ou SVJ, la reward ne reflète donc **pas** le vrai P&L du modèle (le vega P&L dû aux mouvements de σ manque). Pour corriger sous SABR, il faudrait recalculer `V_i` à chaque pas via BS avec `σ_impl_i = sabr_implied_vol(F_i, K, T−t_i, σ_i, ν, ρ)`.

---

## Références

- Cao, J., Chen, J., Hull, J., Poulos, Z. (2021) — *Deep Hedging of Derivatives Using Reinforcement Learning*.
- Hagan, P. S. et al. (2002) — *Managing Smile Risk*.
- Bartlett, B. (2006) — *Hedging Under SABR Model*.
- Fujimoto, S., Meger, D., Precup, D. (2019) — *Off-Policy Deep Reinforcement Learning without Exploration* (extrapolation error).