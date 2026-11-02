# Reviewed gains and losses from swapping sequence

Ranks are original MPNet20→MiniLM10 versus reverse MiniLM20→MPNet10.
All targets below are eligible; synthetic labels have not been independently
adjudicated. These are qualitative observations, not causal phrase attributions.

- D116, Technical Writers, 2→6: design/layout and organizing information overlap
  with layout work, but technical documentation is not explicit. This is a lost
  top-five target that may reflect broad creative versus technical domain intent.
- D070, Management Analysts, 2→4: routine invoice/inventory record keeping differs
  from organizational studies and efficiency consulting. The label captures broad
  interest compatibility more strongly than explicit consulting intent.
- D038, Marketing Managers, 2→3: selling and leading plausibly support this target,
  while market/product strategy is unstated. The demotion stays in the top five.
- D005, Document Management Specialists, 4→7: records, spreadsheets and procedures
  support the target broadly, but enterprise document systems are more specific.
  This is another lost top-five exact target worth reviewing against accounting roles.
- D018, Survey Researchers, 4→6: data analysis supports research, but survey design
  is not stated; the query also contradicts itself about detail-oriented work.
- D003, Project Management Specialists, 6→1: leading a fundraiser plausibly aligns
  with leading staff and coordinating projects. The reverse sequence promotes a
  supported management interest, without proving seniority or qualification fit.
- D031, Human Resources Managers, 3→1: orientation coordination, schedules and
  helping new students plausibly align with organizing people and staff.
- D019, Artists' Agents and Business Managers, 2→1: sales plus creative projects
  support representation/promotion, though conflicting mechanical preferences
  remain in the raw query and are not explicitly resolved by either model.

Both semantic models still consume raw user history and descriptions only.
All original RIASEC scores and both stage traces are preserved. The experiment
changes which model partitions the first ten candidates and which orders them.
