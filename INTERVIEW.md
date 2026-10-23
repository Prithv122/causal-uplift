# Interview Prep — Causal Uplift

**Five questions, five answers.** An unanswered question means this project is not shipped.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ A simulator draws users with eight covariates and a hidden confounder, and writes down each
user's two conversion probabilities, with and without the offer, so the true individual effect is
exact. Four planted segments: persuadables, sure things, lost causes and sleeping dogs. Two knobs
control how biased the targeting is: `gamma` (assignment depends on an observed variable that also
drives conversion) and `hidden` (part of it moves into a variable no model sees). For each setting
I simulate 10,000 training users, fit eight estimators, then score them on 20,000 fresh users
against the true effect: ATE bias and interval coverage, PEHE, Qini against the truth, and the
value of treating the top 20%. 200 replications per setting, seeded by (setting, replication), run
in parallel, one CSV per setting so a run resumes. A report script turns the CSVs into tables and
applies the pre-registered rules mechanically.

### Q2. Why did you write your own S/T/X/DR-learners instead of using EconML?

_A:_ Because the question is about the methods, and I need to be able to defend every line. Each
learner is under 60 lines on top of scikit-learn or LightGBM models. The tests rebuild each one
independently and compare to 1e-10, and 25 deliberate formula mutations all fail a test. EconML
is still there as a reference arm (its causal forest) behind an optional extra, because a
reviewer will reasonably ask whether my implementation agrees with the standard library. It
pins scikit-learn below 1.10, which is another reason not to make it a core dependency.

### Q3. What's the weakest part of this, and what would break first?

_A:_ The headline numbers come from my own simulator, so they hold for this data-generating
process, not for every business. I chose the effect shapes, and a different shape could favour a
different learner. My reading, not something the grid tested, is that the S-learner ranks best
here partly because the lift is a shift on the logit scale, which a tree model with treatment as a
feature captures easily.
Second, IPW's standard error treats the estimated propensity as known, so its intervals are
conservative. Third, the real-data check on Criteo can only use outcome-based Qini, because real
data has no individual truth. The design choice I'd defend is that the simulator is in the repo,
specified in `DESIGN.md` and tested formula by formula, so anyone can change the shape and re-run.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Three hypotheses were registered before the full run, with exact cells and thresholds. H1,
supported: at random assignment the X- and DR-learner rank users far better than the response
model (true-effect Qini 0.877 and 0.800 vs 0.480; paired gaps +0.397 and +0.320, MC SE about
0.0015). H2, not supported: the naive estimate is biased everywhere from gamma 0.5 (by 0.079 to
0.207 on a true ATE of about 0.079), and AIPW stayed within 0.0011 with 0.93–0.98 coverage, but
IPW with a LightGBM propensity failed at gamma 1 (bias 0.0123 against its own SE of 0.0117,
coverage 0.86). H3, not supported in one of eight conditions, for an interesting reason: LightGBM
IPW's over-correction partly cancelled the hidden-confounding bias. I report the failures as
failures. A review pilot had shown the IPW problem before the run, and I deliberately did not
change H2; I added an exploratory logistic-propensity IPW arm instead, which stayed within 0.005
everywhere. That isolates the cause: the propensity model, not the weighting.

### Q5. Why did the DR-learner have the best ATE but a worse ranking than the S-learner?

_A:_ The DR-learner's ATE is the mean of its pseudo-outcome, so the large individual errors from
inverse-propensity terms average out, and the doubly robust correction removes the confounding
bias. Its CATE comes from regressing that same pseudo-outcome on X, and there the variance does
not average away: a user with a propensity of 0.02 gets a residual multiplied by 50. As
confounding grows, propensities get more extreme, the pseudo-outcome gets noisier, and the
second-stage model fits noise, so Qini fell from 0.800 at gamma 0 to 0.553 at gamma 2. The
S-learner never divides by a propensity; it just compares the model's prediction with t = 1 and
t = 0. That costs it some regularisation bias in the average (-0.0038 at gamma 0), but its ranking
stays stable (0.941 to 0.914). The practical lesson is to use different estimators for "what is
the average effect" and "whom should we treat", and to trim or overlap-weight before fitting a
DR-learner on observational data.

---

## 30-second pitch

I built a simulator where every user's true response to an offer is known, then measured how far
eight causal estimators land from that truth as targeting bias grows. With pre-registered rules:
uplift models beat "target the likeliest buyers" by a wide margin (Qini 0.80–0.94 vs 0.48), AIPW
stayed unbiased under observed confounding, IPW broke at moderate confounding because of its
propensity model, and nothing fixed hidden confounding. Two of my three hypotheses failed, and
the write-up says why.
