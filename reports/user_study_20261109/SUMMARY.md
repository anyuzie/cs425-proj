# Real-user sessions, 9 November 2026

Six people used the live chatbot on 9 Nov 2026, 18:33–18:54 SGT. Source: `logs/anon-*.jsonl`. Screenshots are in `screenshots/`. They were replayed from the logs into the frontend the participants saw (commit `a7e8468`, participant view). Bot replies come from the log and were not regenerated.

The app was running branch `fix/contradiction-question-context` (`a7e8468`) with T5 NLG (`t5-v3`) and the ranker that was then the default. That is not the current `main` ranker.

## Sessions

| Code | Opening message | Time | Bot actions | Top 5 recommendations |
|---|---|---|---|---|
| 5b010d4e | "Architecture and design" | 3m15s | 9 CLARIFY | Architects; Architecture Teachers; Art Directors; Interior Designers; Landscape Architects |
| 4eb8728a | "i studied nursing and i enjoy flexible and easy jobs" | 1m43s | 1 PROBE, 8 CLARIFY | Nursing Instructors; Clinical Nurse Specialists; Nurse Midwives; Registered Nurses; Nurse Practitioners |
| c8fdec4e | "i like doing creative things" | 1m51s | 9 CLARIFY | Art Directors; Art/Drama/Music Teachers; Commercial & Industrial Designers; Special Effects Artists; Video Game Designers |
| fc10207b | "i like flexible jobs" | 1m09s | 9 CLARIFY | Compensation/Benefits Specialists; Industrial-Organizational Psychologists; Compensation & Benefits Managers; HR Specialists; Equal Opportunity Officers |
| 678c2746 | "i studied Aircraft systems engineering, i like collaborative work and having team discussion" | 3m25s | 2 PROBE, 2 ABSTAIN, 5 CLARIFY | Airline Pilots & Flight Engineers; Transportation Inspectors; Teaching Assistants; Biological Technicians; Geological Technicians |
| de5e1065 | "I studied Politics, Law and Economics … non-profit … event planning, team building and volunteer management" | 3m13s | 9 CLARIFY | Sociologists; Political Scientists; Social Work Teachers; Fundraisers; Political Science Teachers |

## Headline numbers (n = 6)

- **Completion:** 6/6 people finished all 10 turns and reached recommendations.
- **Time:** median 2.5 minutes per conversation (range 1m09s–3m25s).
- **Reply length:** 3.1 words per reply on average. 44 of 60 replies (73%) were one or two words ("yes", "no", "maybe", "sure").
- **Dialogue acts:** 3 PROBE in 2 sessions, 2 ABSTAIN in 1 session, 0 CHALLENGE, 1 RECOMMEND per session.
- **Fit to stated field:** in 5/6 sessions the top 5 match the field or interest the person named (architecture, nursing, creative work, aviation, non-profit and social work). This is our own reading; participants did not rate the recommendations. In the sixth session ("i like flexible jobs"), the person gave no field, and the list is HR and compensation roles.

## What worked

- Probing fired on the familiarity cases it was designed for. In "I studied nursing…" the bot asked whether helping people is enjoyed or just known. In the aircraft session it asked twice: about practical work, and about research ("i hate research").
- Abstain worked as designed: after "not really but doable" and "it feels alright", the bot rephrased instead of guessing.
- Recommendations used the person's own words. Field-specific openings ("nursing", "architecture", "aircraft") led to field-specific jobs.

## What didn't work

- **Real users reply in one or two words.** That leaves the extractor and the contradiction check little to work with, so most of the profile came from yes/no answers to bank questions.
- **Repeated questions:** 2 sessions were asked the same question twice ("pitching ideas and persuading people" twice; "leading a team and making decisions" twice).
- **A contradiction was missed.** In de5e1065 the person said "I would love being part of the team, but I do not enjoy making decisions", then later answered "yes" to "leading a team and making decisions". No CHALLENGE fired. This is the known gap where one message holds two stances.
- **The first question ignores the opening.** After "Architecture and design" the bot asked about "helping or teaching people". The opening is read, but the first question is not chosen from it.

## Caveats

- Six participants, recruited informally, with no rating data in the logs. "Rate these five" opens an external form, so collect those responses separately if there are any.
- The sessions ran the previous ranker and code. The current `main` (PR #17, #18) changes the ranker and frontend.
