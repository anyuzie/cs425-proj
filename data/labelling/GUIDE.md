# Labelling real student sentences: team guide

Goal: about 200–300 real sentences. Each member collects 40–60 sentences from 5–10 classmates.

## 1. Collecting
- Ask: **"What kind of work or activities do you enjoy, and why?"** and **"What kind of work would you hate doing?"** Follow up once with one CLARIFY question from question_bank.json and copy the reply.
- Copy replies **word for word**: keep typos, Singlish and "idk". Split long replies into sentences.
- Get consent and don't record names. Use a participant code such as `CH03`.

## 2. Labelling (one row per sentence, in real_sentences_template.csv)
| Column | What to write |
|---|---|
| `text` | The sentence, exactly as said |
| `context` | The bot's question if this is a reply to one, otherwise empty |
| `R` … `C` | For each type: `+` like, `-` dislike, or empty (no stance) |
| `R_str` … `C_str` | For each marked type: `strong`, `mild` or `dontmind` |
| `reason` | `enjoyment`, `familiarity` or `neither` |
| `labeller` | Your initials |

## 3. The six types
- **R Realistic:** hands-on work, tools, machines, building, outdoors, animals, sports, driving
- **I Investigative:** research, science, analysis, maths, data, coding, figuring out why
- **A Artistic:** art, design, writing, music, film, media, performing, creative cooking
- **S Social:** teaching, helping, caring, counselling, healthcare, customer service, HR
- **E Enterprising:** business, selling, marketing, leading, managing, persuading, law, politics, finance
- **C Conventional:** records, accounting, admin, data entry, procedures, organising, scheduling, routine

O*NET's own word list per type is in `riasec_lexicon.json`. Use it as a reminder, not as a rule.

## 4. Strength rules (same rules as the training data)
- **strong:** intensifiers or strong words: *really, so, super, absolutely, love, hate, can't stand, dread, nightmare, for hours, favourite, exciting, amazing*
- **mild:** hedges or plain statements: *kinda, sort of, I guess, probably, a bit, okay, not really my thing, I like X, X sounds boring*
- **dontmind:** *either is fine, don't mind, no preference*. This is a weak positive.
- If a sentence has both a hedge and an intensifier, the hedge wins and you label it **mild**.
- *Not sure, depends, idk* → **no stance**. Leave the type empty and set reason `neither`.

## 5. Reason rules
- **familiarity:** the lean comes from course, family, job, being told they're good at it, or "it's safe", e.g. "I did accounting in poly so I guess accounting". Still mark the type `+` or `-`.
- **enjoyment:** they talk about liking or disliking the activity itself.
- If both apply ("my dad's an engineer and honestly I love building stuff"), choose **enjoyment**.

## 6. Two labellers, then agree
Each sentence is labelled by two people independently. Run `python import_real.py your_file.csv`, which prints the agreement per column. Discuss every disagreement and keep the agreed label. Report the agreement figure (Cohen's kappa) in the paper.

## 7. Where the sentences go
`import_real.py` splits the sentences: **50 go into `real_test`, which you never train on**, and the rest go into train. Report extractor F1 on the synthetic test split and on `real_test`.
