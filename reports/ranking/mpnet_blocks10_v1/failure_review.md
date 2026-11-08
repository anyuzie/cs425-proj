# MPNet versus MiniLM: inspected gains and losses

All listed targets are eligible. Text and target descriptions were inspected,
but relevance labels are synthetic and have not been independently adjudicated.
Ranks below are MiniLM → MPNet within the same baseline block of 10.
No RRF or stance filtering is used, and structured scores remain identical.

## Losses

- D013, Nanotechnology Engineering Technologists and Technicians, 1 → 2:
  analyzing and tinkering plausibly match testing equipment, but nanoscale work
  is not stated. The query also contradicts itself about enjoying research.
  Neither encoder resolves that upstream contradiction explicitly.
- D005, Document Management Specialists, 2 → 6: accurate records, spreadsheets
  and procedures provide plausible support for the target description. The
  shift loses an exact top-five match and warrants review against competing
  accounting occupations; the utterance does not specify enterprise document systems.
- D038, Marketing Managers, 2 → 4: persuading customers and running an online
  shop support commercial interests. Marketing management's market analysis
  and product strategy are more specific than the utterance. This loss is
  within the top five and reduces reciprocal rank without changing Hit@5.
- D048, Meeting, Convention, and Event Planners, 2 → 4: creative projects for
  events and organizing things plausibly support the target, but family-stall
  sales and several dislikes also appear in the query. The rank loss cannot
  be attributed to one phrase without a controlled text ablation.
- D070, Management Analysts, 2 → 4: inventory records, invoices and clear
  procedures resemble routine administrative work more than organizational
  studies and efficiency consulting. A synthetic RIASEC target can therefore
  lose rank despite a reasonable domain distinction by the encoder.

## Gains

- D057, Chief Executives, 9 → 1: leading a fundraising drive and driving ideas
  forward plausibly align with organizational direction. Family expectations
  about practical work remain in the raw history; this gain is not proof of
  readiness for a senior occupation.
- D031, Human Resources Managers, 2 → 1: coordinating orientation, schedules
  and people plausibly match staff coordination. It does not independently
  validate HR as the most appropriate occupation.
- D008, Climate Change Policy Analysts, 4 → 2: research, data and presenting
  findings support general analytical work, but climate policy is not stated.
  Better target rank here need not establish better semantic intent matching.

These are qualitative hypotheses, not causal explanations. Exact profile-level
metric changes and all candidate scores/ranks are preserved in the paired
artifacts. The aggregate evidence supports keeping MPNet experimental.
