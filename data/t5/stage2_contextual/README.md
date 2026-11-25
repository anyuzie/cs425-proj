# T5 wording v2 synthetic dataset

60 background scenarios, split before expanding to 21 examples each: six RIASEC
interests, three wording approaches, three question intents, realistic retries and
additional clarification without known interests. Seed 73; 42 train / 9 validation /
9 test scenario groups. Background groups and exact inputs do not cross splits.
Canonical question subjects and output patterns intentionally occur in all splits;
this tests unseen background combinations, not novel language templates.

Targets are AI-authored and template-expanded, not independently human reviewed.
No participant transcripts, held-out ranking profiles or occupational labels were
used. Prior observed failures inform the new requirements, so the old 112-case suite
is a development regression, not an untouched test. New test rows stay frozen until
the checkpoint is selected by validation loss. Python owns quotes and acknowledgements.
