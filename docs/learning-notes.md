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

---

## Phase 3: Cleaning the data and re-checking the headline number

### 1. Which days count as trading days (the calendar rule)
**Rule:** a Reliance bar is kept only if NIFTY 50 also has a bar that day (the index defines the NSE trading calendar), and zero-volume "flat" bars (Volume = 0 and High = Low) are always dropped. A missing India VIX day does *not* remove the Reliance session; it only blanks the two VIX features for that row and the next.

The rule lives in `align_trading_calendar()` in `src/features/generator.py`. Training, targets and live inference all call it, so they cannot disagree.

**The 11 dates it removes** (the 10 Reliance-only dates found in Phase 2, plus one bad bar):

| Date | What the Reliance bar is | Why it goes |
|---|---|---|
| 2026-01-15, 2026-05-01, 2026-05-28, 2026-06-26, 2026-09-14, 2026-10-02 | Zero-volume flat placeholder; no NIFTY or VIX bar | Market closed; the provider padded the series |
| 2025-03-18 | Zero-volume flat bar on a normal trading day (NIFTY and VIX traded) | Bad data for Reliance that day |
| 2019-10-27 (Sun), 2020-11-14 (Sat) | Real but short Diwali Muhurat sessions; no NIFTY or VIX bar | Not regular sessions; no market context available |
| 2018-01-01, 2019-01-01 | **Real sessions** with normal volume; VIX has a bar but yfinance's NIFTY does not | **Trade-off:** keeping them would blank the 50-day NIFTY window and drop ~50 rows. Dropping them costs one session each |

**Why VIX doesn't define the calendar:** VIX is missing on 17 ordinary trading days (e.g. 2021-01-22). If those days were removed from Reliance, the next day's "1-day return" would really be a 2-day return. NIFTY's gaps are much rarer and are mostly real non-sessions.

### 2. A target bug the cleaning exposed
The label "did the price go up tomorrow?" was computed *after* the feature rows were filtered. So for the row just before a gap, "tomorrow" was the next row that survived. That was 3 sessions later in most cases, and **50 or even 88 sessions later** around holidays. 19 old labels were like this, 5 of them in the test set. The fix computes the next-day return on the trading calendar first, then attaches it to the feature rows. A check confirms every label now equals the true next-session return.

*Interview angle:* "A label that silently means 'two months later' instead of 'tomorrow' would never show up as an error, only as slightly odd results. I found it by asking how many sessions each label actually spans."

### 3. Fixed date boundaries instead of percentages
The split used to be "first 70% of rows / next 15% / last 15%". Cleaning added rows, so percentages would have moved the boundaries and changed the test period. The split now uses fixed dates in `config/settings.yaml`: train ≤ 2023-10-19, validation ≤ 2024-12-20, test 2024-12-23 → 2026-09-10. These are the same boundaries as the old run, so the comparison is fair on dates.

| Split | Old rows | New rows | Why it changed |
|---|---|---|---|
| Train | 1,323 | 1,513 | Post-holiday gaps are filled in |
| Validation | 283 | 283 | No long gaps in this period |
| Test | 284 | 419 | The 2026 holidays had removed ~135 test days |

### 4. Choosing features without looking at the test set
The ablation now trains the MLP on F0…F5 and compares **validation** ROC-AUC only. The test set is never scored during this step, and a test checks that. Validation picked **F5 (all 27 features)**, the same set as before, with F3 second (0.5375 vs 0.5473). So the old choice wasn't wrong. What was wrong was the *evidence* for it.

### 5. One source of truth for every number
`reports/metrics.json` (built by `python main.py metrics`) holds both runs with full provenance: data, split, feature selection, every metric, backtest and artifact paths. The dashboard, the live engine (model, scaler and feature list) and the README all read the run named `active_run`. The README is rendered from `docs/README.template.md`, and a test fails if they drift. Building it exposed more README numbers that no code produces: backtest volatility 11.45% (really 13.77%), Buy & Hold Sharpe 0.27 (really 0.31), the LSTM row's precision/recall/F1, and the train/validation UP shares.

### 6. The result
| Test-set metric | Old run (284 days) | New run (419 days) |
|---|---|---|
| ROC-AUC | 0.5676 | **0.5363** |
| Balanced accuracy | 57.39% | **52.82%** |
| F1 | 0.5399 | **0.4513** |

The headline number fell by about 0.03 ROC-AUC. The new score sits close to the validation score (0.547), while the old one was well *above* its validation score (0.539). That's the pattern you'd expect when a test number has been used to make choices, or was simply a lucky sample. Random Forest (0.532) and XGBoost (0.513) are close behind on test, and both beat the MLP on validation.

*Interview soundbite:* "When I cleaned the data and stopped using the test set for decisions, the headline ROC-AUC dropped from 0.568 to 0.536. I'd rather report the smaller honest number, which is close to chance and close to simple baselines, than the bigger one I can't defend."

---

## Phase 4: Auditing the backtest

### What a backtest has to get right
A backtest answers one question: "If I had followed the model's signals, with realistic trading, would I have done better than just holding the stock?" Four things decide whether the answer is honest:
1. **Timing:** you can only trade *after* you know the signal.
2. **Return period:** each signal earns exactly the next period's return, no more.
3. **Costs:** every buy and every sell costs money.
4. **Benchmark:** compare against buy-and-hold over *exactly the same days*.

### Bug: "next row" is not "next day"
This is the same bug as Phase 3's targets. The old engine took the next *prediction row's* close as "tomorrow". Around the data gaps, "tomorrow" was up to 88 sessions later. A single "stay in cash" signal on 2026-01-14 was credited with dodging a −6.1% move that happened over 50 sessions. It also distorted buy-and-hold, which showed +5.1% when the same days actually gave +16.4%. The fix computes returns on the trading calendar first, then lines them up with the predictions.

### Two execution assumptions
- **Close-to-close (optimistic):** the signal is computed from today's close, and you trade *at* that same close. In practice you can't, because the close is only known once trading has finished.
- **Next open (executable, now the headline):** you place the order after the close and get filled at tomorrow's open, holding until the following open.

The model was trained to predict close-to-close moves. If its edge disappears at the next open, the signal isn't tradeable.

### Costs, stated precisely
0.15% **per side** (0.10% brokerage/fees + 0.05% slippage), charged on every entry and every exit, including closing the position at the end. A round trip costs 0.30%. Buy-and-hold pays one entry and one exit too.

### Sharpe ratio and the risk-free rate
Sharpe = average *excess* daily return over the risk-free rate, divided by its volatility, ×√252. I used a constant **6.0% a year**. That's an *assumption* approximating the RBI 91-day Treasury-bill yield over the test period (auction cut-offs roughly 5.5–6.7%). I couldn't get a verified monthly series, so Sharpe is also reported at 0%, 5% and 7%. The conclusion is the same at every rate.

### Two win rates
- **Per day:** share of in-market days that made money.
- **Per trade:** share of complete round trips (entry → exit, after both costs) that made money. This is what a trader experiences. A trade can have more winning days than losing ones and still lose money after a 0.30% round-trip cost.

### Results (new model, 419 test days, 0.15% per side)
| | Next open (headline) | Close-to-close |
|---|---|---|
| Strategy | **−17.1%** | −6.5% |
| Buy & hold, same days | −1.4% | −0.8% |
| Sharpe (rf 6%) | −1.11 vs −0.16 | −0.73 vs −0.20 |
| Win rate per trade / per day | 32.8% / 43.4% | 31.1% / 48.0% |
| Trades / costs paid | 61 / 18.3% of capital | 61 / 18.3% |

**Where the loss comes from:** with *zero* costs and close-to-close execution, the strategy made +12.3% against −0.5%, so there is some gross signal in the optimistic setting. Two things remove it:
1. **Costs.** 61 round trips at 0.30% each cost about 18% of capital.
2. **Execution timing.** At the next open, even the zero-cost strategy only breaks even (−0.4% vs −1.1%). The overnight move from close to open carries much of what the model "knew".

### Why the old +23.8% looked so good
Re-running the old model's predictions through the fixed engine gives +18.9% vs +16.4% (close-to-close) and +14.8% vs +15.7% (next open). The old headline compared an inflated strategy return against a buy-and-hold figure the bug had pushed down. And those numbers only covered 283 of 425 sessions.

### Interview soundbite
"After fixing the backtest and using realistic execution and costs, the strategy lost 17% while buy-and-hold lost 1.4% over the same days. The model has a little gross signal close-to-close, but it trades too often (73× turnover a year) and the edge doesn't survive to the next day's open. That's a negative result, and it's the most useful finding in the project: classification accuracy just above 50% doesn't turn into a profitable daily strategy once costs and timing are real."

---

## Phase 5: How sure are we? (uncertainty)

### Why a single number isn't enough
A test ROC-AUC of 0.536 on 419 days could easily come out at 0.50 or 0.58 on another 419 days. A **confidence interval** gives the range of scores that would plausibly appear if we could re-run history many times.

### Why a *block* bootstrap
An ordinary bootstrap resamples days independently, as if they were unrelated. Here they aren't: the MLP's probability today is strongly linked to yesterday's (autocorrelation 0.49). Resampling single days would make the results look more certain than they are. The **stationary block bootstrap** resamples *runs* of consecutive days (random lengths, 10 on average). That keeps the day-to-day dependence. I checked block lengths of 5 and 20 too, and the intervals barely move.

**Paired comparison:** every model is scored on the *same* resampled days. That makes "MLP minus Random Forest" a fair difference: both face the same easy and hard stretches.

**Second check against chance (circular shift):** slide the labels against the predictions by 20+ days. That keeps both series' patterns but destroys any real link between them. If the real score isn't clearly better than the shifted scores, there's no evidence of skill.

### Results (419 test days)
| Model | ROC-AUC | 95% CI | Chance-test p |
|---|---|---|---|
| MLP | 0.536 | 0.481 – 0.591 | 0.09 |
| Random Forest | 0.532 | 0.475 – 0.593 | 0.13 |
| XGBoost | 0.513 | 0.466 – 0.564 | 0.34 |

MLP − Random Forest: +0.005 (CI −0.052 to +0.060).

**Plain reading:** every interval contains 0.5, so on this test set **none of the models is distinguishable from random guessing**. The MLP is **not** distinguishable from Random Forest or XGBoost either. Balanced accuracy tells the same story.

### Can a no-trade band rescue the strategy?
Idea: trade less. Go long only when the model is confident (P(up) ≥ upper), go to cash only when it's confident the other way (P(up) < lower), and otherwise keep the current position. Fewer trades means lower costs.
- The requested 0.55 / 0.45 band **never trades on validation**, because the model's validation probabilities only range from 0.469 to 0.545. The model is never that confident. So I also tested narrower bands.
- The band was chosen on **validation**, then the test set was evaluated **once**: ±0.02 (0.52 / 0.48).
- Validation: **no band beat buy-and-hold** (best −3.2% behind).
- Test: −0.16% vs −1.43% for buy-and-hold, with only **9 trades** and costs cut from 18.3% to 2.7%. Sharpe was still below buy-and-hold's.

So the band fixes the cost problem but not the signal problem. 9 trades, chosen from a jumpy validation grid, are not evidence of a working strategy.

### Interview soundbite
"I put block-bootstrap confidence intervals on the test metrics because daily predictions are autocorrelated. Every interval included 0.5, and the MLP couldn't be separated from a Random Forest. A no-trade band cut costs from 18% to under 3%, but it never beat buy-and-hold on validation. The honest conclusion is that I couldn't find a reliable, tradeable next-day signal for Reliance with these features."

---

## Phase 6: Do sequence models (LSTM, GRU) help?

### The idea
The MLP sees one day at a time: the 27 features for today. An **LSTM** or **GRU** reads the last *N* days in order, so in principle it can pick up patterns that unfold over time. A GRU is a slightly simpler cousin of the LSTM, with fewer parameters.

### Keeping the comparison fair
- **Decided in advance:** LSTM and GRU, each with a 10-day and a 20-day lookback. Same small architecture for all four, no tuning.
- **Same inputs and splits:** same 27 features, same fixed date boundaries, scaler fitted on training rows only.
- **Early stopping on validation only.** Each model scored the test set exactly once.
- **Same test days:** each window ends on the day it predicts (earlier days in the window are inputs only), so all models are judged on the same 419 days. That makes the bootstrap differences "paired".

### Results
| Model | Val ROC-AUC | Test ROC-AUC (95% CI) | vs MLP |
|---|---|---|---|
| MLP | 0.547 | 0.536 (0.481–0.591) | — |
| LSTM 10d | 0.508 | 0.503 (0.457–0.549) | −0.034, CI includes 0 |
| LSTM 20d | 0.516 | 0.502 (0.455–0.547) | −0.035, CI includes 0 |
| GRU 10d | 0.511 | 0.533 (0.487–0.579) | −0.003, CI includes 0 |
| GRU 20d | 0.511 | 0.508 (0.462–0.553) | −0.028, CI includes 0 |

**Plain reading:** none of the sequence models beats the MLP, and none has a ROC-AUC clearly above chance. They're not kept. The README records this as a negative result.

### The tempting false positive
GRU 10d's *balanced accuracy* interval on test (50.4–58.6%) just excludes 50%. It's tempting to say "the GRU works!". Three reasons not to:
1. On **validation** it scored 47.5% balanced accuracy, *worse than a coin*. A real signal would show up in both periods.
2. **Multiple testing:** 14 intervals have been checked against 0.5 so far. At 95% confidence, about 0.7 of them should "succeed" by luck even if no model has any skill. After a Bonferroni correction (0.05 / 14 ≈ 0.004), its p = 0.018 isn't significant.
3. Switching to it *because* of its test score would be choosing a model on the test set, the exact mistake fixed in Phase 3.

### Why more complex models didn't help
With about 1,500 training days and a signal this weak, a model that can represent more patterns mainly learns more noise. Early stopping kicked in after 11–22 epochs, a sign that validation loss stopped improving almost immediately.

### Interview soundbite
"I compared LSTM and GRU against the MLP under identical conditions: same features, same splits, early stopping on validation, test scored once. None of them beat the MLP, and none was clearly above chance. One GRU's test balanced accuracy looked significant, but it was below chance on validation and doesn't survive a multiple-comparison correction, so I reported it as noise rather than cherry-picking it."

---

## Phase 7: How confident is the model? (calibration)

### What calibration means
If a model says "60% chance of UP" on 100 different days, about 60 of those days should actually go up. A **reliability diagram** checks this: group days by predicted probability and plot the predicted value against the share of days that really went up. Perfect calibration falls on the diagonal.

Two summary scores:
- **Brier score:** average of (probability − outcome)². Always saying 0.5 scores 0.25, and lower is better.
- **Log loss:** punishes confident wrong answers heavily. A coin flip scores ln 2 ≈ 0.693.

Both mean little on their own, so I compared them with a **no-skill forecast** that always predicts the training share of UP days (51.4%).

### What I found
- On validation, **every** probability fell between 0.469 and 0.545. On test, 98.8% were between 0.45 and 0.55.
- Brier: 0.2500 for the model vs 0.2504 for the constant forecast on test, a *0.16%* improvement. Log loss: 0.6931, the same as a coin flip.
- The reliability bins sit on a thin vertical strip around 0.5, and their 95% intervals (about ±10 points each) overlap one another and the diagonal.

### What it means, plainly
The model is **barely confident**, and that's the *right* behaviour for a model with almost no signal. It has learned that next-day direction is close to a coin flip, so it hedges near 50%. The probabilities aren't wrong so much as nearly uninformative. **Recalibrating them wouldn't create signal**, so no calibration was fitted. This also explains Phase 5: the 0.55/0.45 band never traded because the model is never that sure.

### Interview soundbite
"The reliability diagram was the clearest picture of the whole project: every prediction sits between about 0.47 and 0.55, and the Brier score is 0.2500 against 0.2504 for always predicting the base rate. The model isn't miscalibrated, it's honest about having almost nothing to say. That's consistent with the bootstrap and the backtest."

---

## Phase 8: Presenting it honestly

### The README now leads with the answer
A reader sees the conclusion first ("no reliable, tradeable next-day signal was found"), then one results table with confidence intervals, then the 7 bugs, then the method and the details. Every number is still generated from `reports/metrics.json`. The README is rendered from `docs/README.template.md`, and a test fails if the two drift apart. Rebuilding it also caught four feature formulas that the old README described wrongly (e.g. volatility "annualized", which the code never does).

### Two display fixes on the dashboard
- **Banners:** "Research prototype, not investment advice", plus the plain backtest result, at the top of the Overview.
- **The price card mixed sources.** It showed the live price next to an old session's date and an old day's change. Now price, change and date always come from the same place: either the live quote (compared with the last completed session before it) or the stored history.

### A deployment lesson
After a push, the live server kept running *old* copies of the `src/` modules, because Streamlit only reloads files inside the app's own folder. First the live button failed, then the whole page crashed: new code asked an old cached module for a function it didn't have. A reboot fixes it once. A small guard in `app.py`, which drops cached `src.*` modules at the start of each run, prevents it for good. I tested the guard by planting a stale module on purpose.

*Interview angle:* "Tests passing locally isn't the same as the deployed app working. I checked the public app with a headless browser after every push, and that's how I caught a stale-module crash that never appears locally."
