# StockSense Learning Notes

Plain-English notes on each phase of the audit. They explain what was done, why, and how to talk about it in an interview.

---

## Phase 1: Reconnaissance (mapping the project)

### What I did
I read the whole codebase before changing anything, and traced every number the app displays back to the file and the run that produced it.

### How the project fits together
The project is a chain of steps, each started from `main.py`:

1. **ingest** downloads daily prices for Reliance, NIFTY 50 and India VIX into `data/raw/`.
2. **features** turns those prices into 27 indicators, such as "how far is the price from its 200-day average?". The code is in `src/features/generator.py`.
3. **targets** labels each day: did the price go up the next day (1) or not (0)? The code is in `src/targets/generator.py`. Together, these two steps produce `data/processed/RELIANCE.NS_X.csv` (inputs) and `..._y.csv` (answers).
4. **train** fits a model on the first 70% of days, uses the next 15% to decide when to stop training, and scores the model once on the last 15% (the test set).
5. **backtest** pretends to trade on the test-set predictions and measures profit.
6. **The dashboard** (`app/dashboard/app.py`) reads the saved files. Its "live prediction" button re-runs steps 1–2 on fresh data and feeds the result to the saved model.

### Key things I found
- **The live app and training have to compute features identically.** Both call the same `FeatureGenerator`, which is good. But the live code passed the market data under different names (`"nifty"` instead of `"NIFTY"`), so the six market features were never built. That was the error on the website.
- **The dashboard's 0.5676 ROC-AUC is real and traceable.** It is the test-set score of training run `20261007_003042`. That run's model file is byte-for-byte identical to the model the app loads.
- **The other benchmark (0.5370) can't be traced.** No file or code in the repository produces it. The walk-forward figures in the README (e.g. 0.5491) also have no code that generates them.
- **The README's baseline table mixes up numbers.** For example, the Logistic Regression "test" ROC-AUC (0.5143) is actually the walk-forward average. The saved test result is 0.5070.
- **The test set was looked at during feature selection.** The ablation study compared feature groups on the test set, and the final feature set was chosen because it scored best there. So the test score is optimistic: it was used to make a decision, so it is no longer a pure "unseen exam".
- **Some baseline files are left over from an older 2-year dataset** (44 test days instead of 284), so they can't be compared directly.

### Interview soundbite
"Before touching the code I traced every reported metric to the exact run and file that produced it. That showed one live-pipeline bug, one headline number I could verify, one I couldn't, and a subtle problem: the test set had been used for feature selection, so the headline score is likely optimistic."

---

## Phase 2: Fixing the live prediction bug

### The bug, in one sentence
The live app passed the market data (NIFTY and VIX) to the feature code under lowercase names, but the feature code only looks for uppercase names. It quietly skipped all six market features, and the safety check then refused to predict.

### Why it's a good bug to talk about
- **It was silent.** Nothing crashed. The feature code simply didn't find what it was looking for. A careful final check (column names must match the trained model exactly) is what caught it.
- **The fix was small because the design was right.** Training and the live app already used the *same* feature code, so I only had to fix the names, not rewrite any formulas. Rewriting the formulas from memory would have risked small differences, and the model would have received inputs it was never trained on.

### Two more problems I found in the same code
1. **The live app mixed a 1-minute price bar into daily data.** It appended the latest 1-minute candle as if it were a full trading day, so features like "today's high-low range" were calculated from one minute. The model was trained on full days. Now the live price is only *displayed*, and features always come from completed trading days.
2. **Half-finished days were treated as finished.** During market hours, today's daily bar is still changing. The app now uses today's bar only after 16:00 IST (the market closes at 15:30, plus a margin). Before that, it predicts from yesterday's completed session and labels the prediction with that date.

### The honest-failure rule
If the latest completed day can't produce a full feature row, for example because VIX hasn't published that day, the app **refuses to predict and says why**. It does not quietly use an older row or fill the gap with zeros. A wrong prediction that looks current is worse than no prediction.

### Tests I added (and why each exists)
| Test | What it proves |
|---|---|
| Full schema | Live features have all 27 columns in the exact training order |
| **Parity** | For 10 Sep 2026, the live pipeline reproduces the training row exactly. This is the strongest test: same inputs, same numbers |
| Intraday | A deliberately absurd 1-minute bar (High = 5000) has no effect on the features |
| Missing VIX | No VIX data → clear error, no prediction |
| Missing day | NIFTY missing for the latest day → clear error, not yesterday's row |
| Holiday | A holiday placeholder bar → prediction withheld (see below) |

All of them use saved CSV files instead of the internet, so they give the same result every time.

### A data problem the tests uncovered
On NSE holidays, the saved Reliance data contains a fake "flat" bar (no volume, all prices equal). NIFTY and VIX have no row for those days. A 50-day NIFTY average that includes a missing day comes out blank, so **about 50 trading days get dropped after every such holiday**. This has removed 368 days (about 16%) from the training data over the years. Today's live yfinance data doesn't contain these fake bars, so the live app works. But the training data needs cleaning, and the model needs retraining afterwards. That belongs with the Phase 3 retraining.

### Also cleaned up
- **Numbers no code produces were removed** from the README and dashboard: the walk-forward results, the mixed-up baseline rows and the "statistically significant" claim. *Rule: if I can't regenerate a number from the code, I don't report it.*
- **Honest wording:** the dashboard now says the test period was used to choose features, so the score is probably a little optimistic.
- **Old 2-year-dataset files** were moved to `archive/2y_dataset/` with a note, not deleted.
- **A correction to my own Phase 1 report:** I had called the dashboard's 58.68% zero-cost win rate a copy-paste error. Re-running the backtest showed it's correct. With no trading costs, "share of in-market days that made money" is *mathematically the same* as precision ("share of UP predictions that were right"), because the strategy is in the market exactly on the days the model predicts UP. Two numbers looked like a duplication but were the same quantity under two names.

### Interview soundbite
"The live bug was a key-name mismatch, but the more interesting part was what the tests uncovered: a 1-minute bar leaking into daily features, and a holiday data issue that had quietly removed 16% of the training set. I fixed the live path, added parity tests against the training features, and flagged the data issue for retraining rather than hiding it."
