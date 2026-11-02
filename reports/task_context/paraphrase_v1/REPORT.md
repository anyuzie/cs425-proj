# Current BERT: paraphrase diagnostic

Frozen checkpoint, thresholds 0.5. No training or threshold tuning. Expected content labels were authored by the assistant before inference; this is a targeted diagnostic, not an independently reviewed benchmark.

| Group | Correct pairs | Exact match | Task F1 | Context F1 |
| --- | ---: | ---: | ---: | ---: |
| All | 28/80 | 35.0% | 0.564 | 0.458 |
| task | 16/32 | 50.0% | 0.667 | 0.000 |
| context | 12/24 | 50.0% | 0.000 | 0.703 |
| mixed | 0/12 | 0.0% | 0.667 | 0.667 |
| irrelevant | 0/12 | 0.0% | 0.000 | 0.000 |

## Every score

| ID | Group | Text | Task score | Context score | Correct routes |
| --- | --- | --- | ---: | ---: | --- |
| P001 | task | Repair machinery. | 0.9972 | 0.0001 | Yes |
| P002 | task | Repairing machinery. | 0.9972 | 0.0001 | Yes |
| P003 | task | I enjoy repairing machinery. | 0.0195 | 0.9977 | No |
| P004 | task | I dislike repairing machinery. | 0.0114 | 0.9992 | No |
| P005 | task | Maintain medical records. | 0.9972 | 0.0001 | Yes |
| P006 | task | Maintaining medical records. | 0.9972 | 0.0001 | Yes |
| P007 | task | I enjoy maintaining medical records. | 0.0104 | 0.9992 | No |
| P008 | task | I dislike maintaining medical records. | 0.0082 | 0.9995 | No |
| P009 | task | Teach mathematics to children. | 0.9972 | 0.0001 | Yes |
| P010 | task | Teaching mathematics to children. | 0.9971 | 0.0001 | Yes |
| P011 | task | I enjoy teaching mathematics to children. | 0.0068 | 0.9996 | No |
| P012 | task | I dislike teaching mathematics to children. | 0.0050 | 0.9998 | No |
| P013 | task | Care for elderly people. | 0.9972 | 0.0001 | Yes |
| P014 | task | Caring for elderly people. | 0.9971 | 0.0001 | Yes |
| P015 | task | I enjoy caring for elderly people. | 0.0078 | 0.9995 | No |
| P016 | task | I dislike caring for elderly people. | 0.0068 | 0.9996 | No |
| P017 | task | Prepare meals. | 0.9972 | 0.0001 | Yes |
| P018 | task | Preparing meals. | 0.9972 | 0.0001 | Yes |
| P019 | task | I enjoy preparing meals. | 0.0116 | 0.9991 | No |
| P020 | task | I dislike preparing meals. | 0.0055 | 0.9998 | No |
| P021 | task | Plant vegetables. | 0.9972 | 0.0001 | Yes |
| P022 | task | Planting vegetables. | 0.9972 | 0.0001 | Yes |
| P023 | task | I enjoy planting vegetables. | 0.0237 | 0.9966 | No |
| P024 | task | I dislike planting vegetables. | 0.0093 | 0.9994 | No |
| P025 | task | Design logos. | 0.9971 | 0.0001 | Yes |
| P026 | task | Designing logos. | 0.9971 | 0.0001 | Yes |
| P027 | task | I enjoy designing logos. | 0.0073 | 0.9996 | No |
| P028 | task | I dislike designing logos. | 0.0055 | 0.9998 | No |
| P029 | task | Write computer programs. | 0.9973 | 0.0001 | Yes |
| P030 | task | Writing computer programs. | 0.9970 | 0.0001 | Yes |
| P031 | task | I enjoy writing computer programs. | 0.0053 | 0.9998 | No |
| P032 | task | I dislike writing computer programs. | 0.0054 | 0.9998 | No |
| P033 | context | Work outdoors. | 0.9971 | 0.0001 | No |
| P034 | context | An outdoor working environment. | 0.9959 | 0.0002 | No |
| P035 | context | I prefer working outdoors. | 0.0039 | 0.9999 | Yes |
| P036 | context | I dislike working outdoors. | 0.0047 | 0.9998 | Yes |
| P037 | context | Work in quiet surroundings. | 0.9972 | 0.0001 | No |
| P038 | context | A quiet workplace. | 0.5219 | 0.5066 | No |
| P039 | context | I prefer a quiet workplace. | 0.0043 | 0.9998 | Yes |
| P040 | context | I dislike a quiet workplace. | 0.0042 | 0.9999 | Yes |
| P041 | context | Work night shifts. | 0.9972 | 0.0001 | No |
| P042 | context | A night-shift schedule. | 0.9933 | 0.0003 | No |
| P043 | context | I prefer working night shifts. | 0.0036 | 0.9999 | Yes |
| P044 | context | I dislike working night shifts. | 0.0041 | 0.9999 | Yes |
| P045 | context | Work as part of a team. | 0.9972 | 0.0001 | No |
| P046 | context | A team-based working environment. | 0.9966 | 0.0001 | No |
| P047 | context | I prefer working as part of a team. | 0.0055 | 0.9998 | Yes |
| P048 | context | I dislike working as part of a team. | 0.0046 | 0.9998 | Yes |
| P049 | context | Work alone. | 0.9967 | 0.0001 | No |
| P050 | context | An environment where people work alone. | 0.9924 | 0.0005 | No |
| P051 | context | I prefer working alone. | 0.0043 | 0.9998 | Yes |
| P052 | context | I dislike working alone. | 0.0050 | 0.9998 | Yes |
| P053 | context | Have freedom to decide how to work. | 0.9971 | 0.0001 | No |
| P054 | context | Freedom to make decisions at work. | 0.9945 | 0.0003 | No |
| P055 | context | I prefer having freedom to decide how I work. | 0.0048 | 0.9998 | Yes |
| P056 | context | I dislike having freedom to decide how I work. | 0.0048 | 0.9998 | Yes |
| P057 | mixed | Repair machinery outdoors. | 0.9972 | 0.0001 | No |
| P058 | mixed | Repairing machinery outdoors. | 0.9972 | 0.0001 | No |
| P059 | mixed | I enjoy repairing machinery outdoors. | 0.0100 | 0.9992 | No |
| P060 | mixed | I dislike repairing machinery outdoors. | 0.0090 | 0.9995 | No |
| P061 | mixed | Teach mathematics in a noisy classroom. | 0.9973 | 0.0001 | No |
| P062 | mixed | Teaching mathematics in a noisy classroom. | 0.9970 | 0.0001 | No |
| P063 | mixed | I enjoy teaching mathematics in a noisy classroom. | 0.0046 | 0.9999 | No |
| P064 | mixed | I dislike teaching mathematics in a noisy classroom. | 0.0046 | 0.9999 | No |
| P065 | mixed | Grow plants in a greenhouse. | 0.9973 | 0.0001 | No |
| P066 | mixed | Growing plants in a greenhouse. | 0.9967 | 0.0001 | No |
| P067 | mixed | I enjoy growing plants in a greenhouse. | 0.0059 | 0.9997 | No |
| P068 | mixed | I dislike growing plants in a greenhouse. | 0.0056 | 0.9998 | No |
| P069 | irrelevant | I am not sure yet. | 0.0042 | 0.9998 | No |
| P070 | irrelevant | I haven't decided yet. | 0.0041 | 0.9999 | No |
| P071 | irrelevant | I don't know what I want yet. | 0.0046 | 0.9998 | No |
| P072 | irrelevant | I am still figuring things out. | 0.0041 | 0.9999 | No |
| P073 | irrelevant | A commercial driver's license is required. | 0.0048 | 0.9998 | No |
| P074 | irrelevant | This occupation requires a commercial driver's license. | 0.0039 | 0.9999 | No |
| P075 | irrelevant | You need a commercial driver's license for this occupation. | 0.0044 | 0.9998 | No |
| P076 | irrelevant | A commercial driver's license is a requirement for this role. | 0.0045 | 0.9998 | No |
| P077 | irrelevant | Hello. | 0.0043 | 0.9999 | No |
| P078 | irrelevant | Hi there. | 0.0046 | 0.9998 | No |
| P079 | irrelevant | Good morning. | 0.0041 | 0.9999 | No |
| P080 | irrelevant | Thanks for your help. | 0.0054 | 0.9998 | No |

## Interpretation

Task/context content should remain unchanged when adding likes/dislikes. Irrelevant text should activate neither route; mixed content should activate both. These are routing diagnostics, not additional model classes.
Exact normalized training-text matches: 0. See summary.json for metrics excluding those matches and family-level routing changes.
Scores are sigmoid outputs, not calibrated probabilities. High confidence on a wrong route is a generalisation failure, not evidence of correctness.
Checkpoint weights were fingerprinted before and after inference and remained unchanged. Existing dataset files, labels and ranking behaviour were not modified.
