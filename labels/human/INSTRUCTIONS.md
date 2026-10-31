# Labelling guide (one page)

Full rules: codebook/codebook.md (v2)

You will label 100 sentences from a monetary-policy committee's post-meeting statements and meeting minutes. The committee sets a short-term policy interest rate, can buy or sell securities (its balance sheet), and aims at maximum employment and inflation of about 2 percent. For each sentence record what kind of statement it is (**FUNCTION**), what it is about (**TOPIC**) and which way it points for the policy rate (**DIRECTION**).

**How the 100 were chosen.** They were drawn at random from inside the 350-sentence validation set (stratified by document type and by whether the sentence contains policy keywords), so your labels can be compared directly with that set's labels. You see no other labels.

**Rules of the task.** Label the `text` column only. `prev_text` is the sentence before it; use it only to resolve words like "this", "they", "such purchases". Names and dates are masked. Use no outside knowledge: do not guess when it was written or what happened later. Judge each sentence on its own words.

## FUNCTION (one per aspect)
- **ASSESSMENT**: current or recent state of the economy or financial conditions (past or present tense; "has risen", "data indicated").
- **OUTLOOK**: projections, expectations or risks about the future ("is expected to", "projected", "risks to the outlook"), whoever holds them.
- **POLICY**: decisions and views about policy itself: rate decisions, balance sheet, forward guidance, how policy will react, dissents, participants' views on what policy should be. "The Committee expects to keep the rate unchanged" is POLICY, not OUTLOOK. Sentences that only describe how policy will be decided ("will monitor incoming information", "as appropriate") and operations to keep the rate in its range or reserves ample are POLICY with direction 0.
- **PROCEDURAL**: no stance: stated long-run goals or mandate in general terms, attendance, voting lists, approval of minutes, logistics, descriptions of documents, headings and fragments. Always topic blank, direction 0.
- Observed short-term rate conditions and facility usage are ASSESSMENT / FINANCIAL / 0; what the committee or its desk does is POLICY.

## TOPIC (ASSESSMENT and OUTLOOK only; blank for POLICY and PROCEDURAL)
- **INFLATION**: prices, inflation measures and expectations, energy and commodity prices, cost pressures.
- **LABOR**: employment, unemployment, payrolls, slack or tightness, wages, productivity.
- **ACTIVITY**: output and growth ("the economy", "the expansion", "demand"), spending, investment, housing, trade, sentiment.
- **FINANCIAL**: asset prices, interest rates on loans and securities, credit, banks, funding strains, financial stability, the dollar.
- **GENERAL**: only the balance of risks, overall uncertainty, or both goals at once.

## DIRECTION: -2, -1, 0, +1, +2
Question: taken on its own, does the sentence point toward a higher (+, hawkish) or lower (-, dovish) future path for the policy rate? Judge the sentence alone, not against earlier statements.
- **INFLATION**: high, rising, above goal, rising expectations = +; low, falling, below goal = -.
- **LABOR**: tight market, strong gains, low or falling unemployment, fast wages = +; weak market, job losses, high unemployment, slack = -.
- **ACTIVITY**: strong or accelerating = +; weak, slowing, falling exports, below potential = -. Imports alone = 0.
- **FINANCIAL**: easing conditions, rising asset prices, narrow spreads, weaker dollar = +; financial stress, falling prices, wide spreads, tighter credit, stronger dollar = -.
- **GENERAL**: upside inflation risks or overheating = +; downside risks to growth or employment, uncertainty that weighs on the outlook = -.
- **POLICY**: raising the rate, signalling increases, reducing the balance sheet or total purchases, removing accommodation = +; cutting, signalling cuts or a long period of low rates, expanding the balance sheet or total purchases, patience about raising, adding accommodation = -.
- Zero: holding the rate with no guidance; outcomes at or near the goals; a bare number with no evaluative word; reporting changes in rates or investor expectations; changing only the pace of purchases with the total unchanged; caution or patience with no stated direction.
- **Strength**: 2 = strong, emphasized or a definite decision ("sharply", "substantially", "elevated", "decided to raise/lower"); 1 = mild or partial ("somewhat", "modestly", "solid", "gradual", "low"); 0 = stable, balanced, mixed or no implication. Risks "tilted" or "skewed" to one side = 1; "significant", "substantial" risks = 2.
- **Hedges and minority views**: lower strength by one, never below 1 (keep the sign), for "may / might / could / some risk", conditional statements, and "a few / some / several / one participant / a member". "Most / many / all participants" and "the Committee" are not hedged.
- **Dissents**: POLICY, direction the dissenter wanted (higher rate +, lower rate -), strength 1. A dissent not about the stance of policy is 0.

## Mixed sentences
Use a second aspect only when the sentence makes two points that differ in FUNCTION or TOPIC ("Inflation picked up, but unemployment rose" = ASSESSMENT/INFLATION/+1 and ASSESSMENT/LABOR/-1). Never two aspects with the same function and topic: give one aspect with the net direction (offsetting or "mixed" = 0; one point clearly dominates = its sign, strength 1; in "Although A, B" the main clause B usually dominates). Background or premise clauses are not separate aspects.

## Anchors
- ASSESSMENT: "Inflation remains elevated." +2 · "Job gains have been solid." +1 · "Economic activity has been expanding at a moderate pace." 0 · "Business investment has softened." -1 · "The unemployment rate has risen sharply and job losses have mounted." -2
- OUTLOOK: "Participants expected inflation to rise well above 2 percent." +2 · "Inflation is expected to rise to 2 percent over the medium term." +1 · "The risks to the outlook are roughly balanced." 0 · "Some participants noted that inflation could fall further." -1 (minority, hedged)
- POLICY: "The Committee decided to raise the target range." +2 · "Further increases in the target range may be appropriate." +1 · "The Committee decided to maintain the target range." 0 · "The Committee can be patient in beginning to normalize the stance of monetary policy." -1 · "The Committee decided to lower the target range." -2
- PROCEDURAL: "The minutes of the meeting held on [DATE] were approved." (topic blank, 0)

## Filling the sheet (`human_sheet.csv`, open in Excel)
Fill one row per sentence; leave the `id`, `prev_text` and `text` columns untouched. Type plain values only.

| Column | Allowed values |
|---|---|
| `function_1` | ASSESSMENT, OUTLOOK, POLICY, PROCEDURAL |
| `topic_1` | INFLATION, LABOR, ACTIVITY, FINANCIAL, GENERAL; blank for POLICY and PROCEDURAL |
| `direction_1` | -2, -1, 0, 1, 2 (`+1` also accepted); always 0 for PROCEDURAL |
| `function_2`, `topic_2`, `direction_2` | the optional second aspect, same values; leave all three blank if there is none |
| `confidence` | `high` (follows directly from the rules), `med` (two reasonable labels), `low` (unclear or not covered) |
| `notes` | optional free text |

If there are two aspects, keep the two most important points and put the more important first. Capital letters do not matter. Do not skip `confidence`. A row with every label cell blank counts as not yet labelled, so you can stop and resume. Save as CSV (keep the file name) after each session.
