# Codebook for labelling central-bank policy-committee sentences

**Version v2** (frozen 2026-10-30)

You will label sentences taken from the written communications of a monetary-policy
committee: its post-meeting statements and the minutes of its meetings. The
committee sets a short-term **policy interest rate** and can also buy or sell
securities (its **balance sheet**). Its goals are maximum employment and stable
prices (an inflation rate of about 2 percent).

For each sentence you record **what kind of statement it is** (FUNCTION), **what it is
about** (TOPIC) and **which way it points for the policy rate** (DIRECTION).

---

## 1. What you see and what you label

Each item has:
- `text`: the **target sentence**. Label this sentence only.
- `prev_text`: the sentence just before it in the same document, or empty. Use it
  **only** to work out what the target sentence refers to (for example "This", "they",
  "such purchases"). Do not label it.

Names, dates and years have been removed or replaced (for example `[DATE]`,
`[NAME]`). **Use no outside knowledge.** Do not try to work out when a sentence was
written, who said it, or what happened afterwards. Judge each sentence only from its
own words (plus `prev_text` for references).

---

## 2. FUNCTION (exactly one per aspect)

| Function | What it covers | Typical cues |
|---|---|---|
| `ASSESSMENT` | The **current or recent** state of the economy or of financial conditions, based on information already observed. | Past or present tense: "has risen", "remained", "was", "data indicated" |
| `OUTLOOK` | **Projections, expectations or risks** about the future economy or financial conditions, held by the committee, its participants or its staff. | "is expected to", "projected", "anticipate that inflation will", "risks to the outlook", "uncertainty" |
| `POLICY` | **Decisions and views about policy itself**: rate decisions, the balance sheet, forward guidance about future policy, how policy will react, dissents, and participants' views on what policy should be. | "decided to", "will maintain", "it would be appropriate to", "the stance of policy", "purchases", "preferred to" |
| `PROCEDURAL` | No stance at all: mandate boilerplate, attendance, voting mechanics and lists of voters, approval of minutes, staff logistics, descriptions of documents, meeting scheduling, administrative actions. | "The Committee seeks to foster…" (stated goals), "Voting for this action were…", "The minutes of the previous meeting were approved", "The meeting adjourned" |

Rules for choosing:
- Describing the economy as it **is or was** → ASSESSMENT. Describing what it **will
  be or might be** → OUTLOOK. If a sentence does both, see Mixed sentences (section 5).
- Anything about **what the committee does or should do with its tools** → POLICY,
  even when it is phrased as a forecast ("the Committee expects to keep the rate
  unchanged" is POLICY, not OUTLOOK).
- **Who** is speaking does not change the function. A staff forecast is OUTLOOK; a
  staff summary of recent data is ASSESSMENT.
- A sentence stating the committee's long-run goals or strategy in general terms, with
  no view on current conditions or the current direction of policy, is PROCEDURAL.
- A sentence that only describes how policy will be decided ("will take into account a
  wide range of information", "will continue to monitor incoming information", "is
  attentive to the risks to both sides of its dual mandate", "will adjust policy as
  appropriate", "is prepared to use its tools as appropriate") is POLICY with direction 0.
- **Implementation and operations** — operations to keep the policy rate in its target
  range, to keep reserves ample, to support the smooth functioning of money markets, the
  mechanics, framework or communication plan of balance-sheet operations, reinvestment
  mechanics — are POLICY with direction 0, unless the sentence says the action adds or
  removes policy accommodation or changes the size of securities holdings to ease or
  tighten.
- Recommendations, preferences or reports by staff or by the operations manager about
  operational details (collateral, counterparties, technical authorities) are PROCEDURAL.
- **Observed money-market conditions** — where the policy rate or other short-term rates
  traded relative to the target, usage of the committee's facilities and repo
  operations, and the reasons for changes in them — are ASSESSMENT/FINANCIAL with
  direction 0. What the committee or its operations desk **does or decides** is POLICY.
- **Arguments about a policy choice** ("a significant delay in raising the rate risked a
  buildup of inflation pressures") are POLICY, with the direction of the policy the
  argument supports.
- Summaries of staff **research or model analysis** that do not describe current
  conditions, the outlook or the stance of policy are PROCEDURAL.
- Section headings, footnote markers and fragments with no content are PROCEDURAL.

---

## 3. TOPIC (ASSESSMENT and OUTLOOK only; `null` for POLICY and PROCEDURAL)

| Topic | Covers |
|---|---|
| `INFLATION` | Consumer and producer prices, inflation measures, inflation expectations (surveys or market-based), energy and commodity prices, import prices, cost pressures in general. |
| `LABOR` | Employment, unemployment, job gains, payrolls, hours, participation, job openings, labor-market slack or tightness, wages and compensation, labor productivity. |
| `ACTIVITY` | Output and growth, consumer spending, business investment, housing and house prices, industrial production, inventories, government spending, exports, imports and the trade balance, sentiment and confidence of households and businesses. |
| `FINANCIAL` | Financial conditions: asset prices, interest rates on loans and securities, credit availability and credit spreads, bank lending and bank health, funding strains, financial stability, the exchange value of the dollar, financial spillovers from abroad. |
| `GENERAL` | The balance of risks, overall uncertainty, or both goals at once, with no single area dominant: "the balance of risks", "risks to the outlook", "overall uncertainty". |

Choosing between topics:
- "Economic activity", "economic growth", "the recovery", "the expansion", "output",
  "demand" → **ACTIVITY**, even when the sentence is about the economy broadly.
- "Slack", "resource utilization", "the labor market" → **LABOR**.
- Measures of inflation expectations or inflation compensation, including those derived
  from inflation-indexed securities → **INFLATION**.
- **GENERAL** only for the balance of risks, uncertainty about the outlook, or the two
  goals considered together ("risks to the outlook", "both sides of its mandate").

If a sentence is about a foreign economy, use the topic it is about (usually ACTIVITY
or FINANCIAL) and judge direction by what it implies for the home economy: weaker
growth abroad → weaker demand for home exports → dovish.

---

## 4. DIRECTION (integer −2 … +2)

DIRECTION asks: **taken on its own, does this sentence point toward a higher (+) or a
lower (−) future path for the policy rate?** Positive = **hawkish** (tighter policy),
negative = **dovish** (easier policy), 0 = no implication or balanced.

Direction is **absolute**: judge the sentence by itself. Do not compare it with
earlier statements or guess how it differs from what was said before.

### 4.1 Sign rules by topic

| Topic | Hawkish (+) | Dovish (−) |
|---|---|---|
| INFLATION | Inflation high, rising or above goal; rising inflation expectations; rising energy, commodity or import prices; broad cost pressures | Inflation low, falling or below goal; falling inflation expectations; falling energy or commodity prices; easing cost pressures |
| LABOR | Strong or tight labor market; job gains strong; unemployment falling or low; wages rising quickly; labor shortages | Weak labor market; job losses; unemployment rising or high; slack; slow wage growth. Faster productivity growth is mildly dovish (it eases cost pressures) |
| ACTIVITY | Strong or accelerating growth, spending, investment, housing; rising house prices; output above its sustainable level | Weak, slowing or contracting activity; falling house prices; falling exports / a trade deficit widening because exports fell (it subtracts from growth); output below its sustainable level. Changes in imports alone → 0 |
| FINANCIAL | Easing financial conditions: rising asset prices, narrowing credit spreads, looser credit, rapid credit growth; a weaker dollar | **Financial stress** and tightening conditions: falling asset prices, widening spreads, tighter credit, bank or funding strains, market turmoil, a stronger dollar |
| GENERAL | Upside risks to inflation; an economy described as overheating | Downside risks to growth or employment; elevated uncertainty that weighs on the outlook |
| POLICY | Raising the rate; signalling future increases; reducing the balance sheet or the total amount of purchases; ending purchases early; removing accommodation; firming | Cutting the rate; signalling future cuts or an extended period of low rates; expanding the balance sheet or the total amount of purchases; patience about raising the rate; adding accommodation; easing |

- Merely **reporting** a change in interest rates, or in what investors expect for
  policy, with no statement about financial conditions or the economy → FINANCIAL with
  direction 0.
- Holding the policy rate unchanged, with no guidance about future moves → POLICY 0.
- Readiness to act **in a stated direction** counts ("prepared to provide additional
  accommodation if needed" → −1; "prepared to adjust policy to address inflation
  pressures" → +1, as conditional statements they are hedged); readiness with no
  direction ("as appropriate") → 0.
- Views on the **neutral or longer-run level of the policy rate**: a lower estimate →
  dovish, a higher estimate → hawkish (POLICY).
- Outcomes described as **at or near the goals** (inflation near 2 percent, maximum
  employment, sustained growth) → 0.
- A **number reported without evaluative wording** ("prices rose 2.3 percent over the
  past 12 months") → 0; direction comes from words such as rising, elevated, weak, below.
- **Pace of securities purchases:** slowing or speeding up purchases while the announced
  total stays the same, or because market functioning has improved or worsened → 0;
  changing the total amount or the size of holdings → ± as in the POLICY row.
- **Caution or patience about "adjustments" without a stated direction** → 0; with a
  stated direction it counts ("patient in beginning to raise the rate" → −1).
- PROCEDURAL is always 0.

### 4.2 Strength

| Value | Meaning | Typical wording |
|---|---|---|
| ±2 | Clear, strong or emphasized signal; a definite committee decision or commitment to move | "sharply", "substantially", "significantly", "elevated", "strong", "severe", "deteriorated", "decided to raise/lower", "anticipates that further increases will be appropriate" |
| ±1 | Mild, moderate or partial signal | "somewhat", "modestly", "slightly", "edged up", "solid", "gradual", "low", "picked up" |
| 0 | Unchanged, stable, balanced, mixed in the same aspect, or no implication for the rate | "remained stable", "moderate pace", "roughly balanced", "little changed" |

Risk statements: risks "tilted", "weighted" or "skewed" toward one side, or risks that
have "increased" or "diminished" → ±1; risks described as "significant", "substantial",
"considerable" or "sharply increased" → ±2.

### 4.3 Hedges and minority views
Lower the strength by one, **but never below 1** (keep the sign), when the sentence is:
- hedged: "may", "might", "could", "possibly", "some risk that", "would likely";
- conditional on a future development: "if conditions evolve as expected, … would";
- the view of a minority: "a few / some / several / a couple of / a number of
  participants", "one participant", "a member".

So a hedged +2 becomes +1; a hedged +1 stays +1; a 0 stays 0. Views of "most / many /
all participants", "participants generally" or "the Committee" are not hedged.

**Dissents** are POLICY with the direction the dissenting member wanted (a preference
for a higher rate is +, for a lower rate is −), and as a single member's view they
are hedged: strength 1. A dissent about something other than the stance of policy (for
example, about lending arrangements or communication) has direction 0.

### 4.4 Anchor sentences

ASSESSMENT
- +2 "Inflation remains elevated." · "The labor market has tightened significantly."
- +1 "Job gains have been solid." · "Consumer spending picked up."
- 0 "Economic activity has been expanding at a moderate pace." · "Inflation was little changed."
- −1 "Business investment has softened." · "Inflation has been running somewhat below 2 percent."
- −2 "The unemployment rate has risen sharply and job losses have mounted." · "Strains in short-term funding markets increased substantially."

OUTLOOK
- +2 "Participants expected inflation to rise well above 2 percent over the coming year."
- +1 "The staff projected that growth would strengthen somewhat next year." · "Inflation is expected to rise to 2 percent over the medium term."
- 0 "The Committee judges that the risks to the outlook are roughly balanced."
- −1 "Most participants saw downside risks to economic growth." (majority view, mild) · "Some participants noted that inflation could fall further." (minority, hedged → 1)
- −2 "The economic outlook has deteriorated substantially, and a significant contraction in activity is expected."

POLICY
- +2 "The Committee decided to raise the target range for the policy rate." · "The Committee anticipates that ongoing increases in the target range will be appropriate."
- +1 "Further increases in the target range may be appropriate." · "Policy accommodation can likely be removed at a gradual pace."
- 0 "The Committee decided to maintain the target range." · "The Committee will continue to monitor the implications of incoming information."
- −1 "The Committee can be patient in beginning to normalize the stance of monetary policy." · "A few participants favored a pause in the reduction of the balance sheet."
- −2 "The Committee decided to lower the target range." · "Exceptionally low levels of the policy rate are likely to be warranted for an extended period." · "The Committee will increase its holdings of securities."

PROCEDURAL (always 0, topic null)
- "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run."
- "Voting for the monetary policy action were [NAME], [NAME] and [NAME]."
- "The minutes of the meeting held on [DATE] were approved."

---

## 5. Mixed sentences

A sentence may contain **at most 2 aspects**. Use two aspects only when the sentence
makes two points that differ in **FUNCTION or TOPIC**, for example "Inflation picked
up, but the unemployment rate rose" → `ASSESSMENT/INFLATION/+1` and
`ASSESSMENT/LABOR/−1`. If there are more than two points, keep the two most important.
If one point is minor, use one aspect. A premise or background clause ("Given the
substantial slack, staff anticipated slight downward pressure on inflation") is not a
separate aspect: label the sentence's main point.

**Never give two aspects with the same function and topic.** When a sentence contrasts
points within one function and topic ("household spending grew strongly, while business
investment moderated"; "although activity will remain weak for a time, growth is
expected to resume gradually"), give **one** aspect with the **net** direction:
- the points roughly offset, or the sentence calls the evidence mixed → 0;
- one point clearly dominates → its sign, strength 1. In "Although A, B" sentences the
  main clause B usually dominates the concession A, unless A is clearly stronger.

---

## 6. Confidence

`high`: the labels follow directly from the rules. `med`: you had to choose between two
reasonable labels. `low`: the sentence is unclear, garbled, or not covered by the rules.

---

## 7. Output format

Write one JSON object per line (JSONL), one line per input item, in input order:

```
{"id": "<id from the input>", "aspects": [{"function": "ASSESSMENT", "topic": "INFLATION", "direction": 2}], "confidence": "high"}
```

- `function` ∈ {ASSESSMENT, OUTLOOK, POLICY, PROCEDURAL}
- `topic` ∈ {INFLATION, LABOR, ACTIVITY, FINANCIAL, GENERAL} for ASSESSMENT and
  OUTLOOK; `null` for POLICY and PROCEDURAL
- `direction` ∈ {−2, −1, 0, 1, 2} as a JSON integer; always 0 for PROCEDURAL
- `aspects` has 1 or 2 entries
- `confidence` ∈ {low, med, high}

Output nothing else: no comments, no explanations, no blank lines.
