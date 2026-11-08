# Aggregation versus MiniLM groups of 10: reviewed cases

All listed targets are eligible. Ranks are MiniLM-only → equal-weight aggregation.
Synthetic labels have not been independently reviewed. These are observations,
not causal attributions to phrases or proof of better user recommendations.

- D013, Nanotechnology Engineering Technologists and Technicians, 1→2:
  analyzing and tinkering overlap with equipment testing, but nanoscale intent
  is absent and the query contradicts itself about enjoying analysis/research.
- D025, Special Education Teachers, Middle School, 3→7: helping children who
  need extra support plausibly supports the target. Losing a top-five match
  is potentially harmful; the query does not specify age group or disability.
- D120, Business Intelligence Analysts, 4→7: tentative preferences for records
  and schedules do not explicitly request market intelligence or data analysis.
  Synthetic RIASEC labels may favor a more specific job than the text supports.
- D108, Producers and Directors, 4→6: video editing plausibly supports media
  production, but direction and production responsibilities are not explicit.
  This is a lost top-five target under aggregation.
- D075, Art Therapists, 3→4: creative workshops and helping children support
  the target broadly; the history also contains conflicting record-keeping
  preferences. The demotion stays in the top five.
- D031, Human Resources Managers, 2→1: organizing orientation, people and
  schedules provides plausible support for staff coordination.
- D002, Cardiologists, 5→2: general investigation and explaining ideas do not
  specify cardiovascular medicine. Improved benchmark rank may not indicate
  improved semantic domain intent.
- D057, Chief Executives, 9→5: leadership and driving ideas plausibly match
  organizational direction, while occupational seniority remains unassessed.

The combined model gains five top-five targets and loses four versus MiniLM-only.
The net one-profile gain does not establish an overall improvement. The paired
Hit@5 change interval includes zero, and MRR is slightly lower. Retain the active
weighted-dot-product default; keep aggregation as a selectable experiment.
