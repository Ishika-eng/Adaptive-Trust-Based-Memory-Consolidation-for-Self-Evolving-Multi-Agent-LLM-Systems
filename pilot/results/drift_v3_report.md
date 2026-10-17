## Recurring stream: first-attempt pass rate by phase (%, pooled over seeds [101, 202, 303] )
| condition | A (old) | B (new) | A2 (old again) | memory at end | tokens in / step | model calls / step | seconds / step |
|---|---|---|---|---|---|---|---|
| no_memory | 10.2 | 44.9 | 10.9 | 0 | 81 | 1.00 | 0.1 |
| static_rm | 39.5 | 10.9 | 34.0 | 186 | 726 | 1.71 | 5.1 |
| static_rv | 38.1 | 10.9 | 20.4 | 142 | 749 | 1.76 | 3.7 |
| static_lww | 34.0 | 10.9 | 28.6 | 76 | 753 | 1.74 | 4.5 |
| atmc_v6 | 33.3 | 26.5 | 23.1 | 100 | 1241 | 1.72 | 6.4 |
| rule_mem | 33.3 | 34.0 | 31.3 | 41 | 1248 | 1.67 | 5.7 |

## Registered rules (recurring stream)
R1  phase B, rule_mem vs static_rm: +23.1 pts (CI +15.0 to +31.3, p=3.1e-07, n=147)  -> need >= +10 and CI > 0: MET
R2  phases B+A2, rule_mem vs static_rv: +17.0 pts (CI +11.6 to +22.4, p=5.04e-09, n=294)  -> beats it
R2  phases B+A2, rule_mem vs static_lww: +12.9 pts (CI +6.8 to +18.7, p=5.08e-05, n=294)  -> beats it
R2  overall (beats both by >= 3 with CI > 0): MET
R3  phase A, rule_mem vs static_rm: -6.1 pts (CI -12.2 to +0.0, p=0.0931, n=147)  -> need diff >= -5: NOT MET
R4  end of B: rule_mem wrongly hides 2/67 audited solutions (3.0%, need <= 15%); A2 31.3% vs A 33.3% (need A2 >= A - 10): MET

## Other paired comparisons (rule_mem minus rival), by phase
- vs no_memory: A: +23.1 pts (CI +15.0 to +32.0, p=3.1e-07, n=147); B: -10.9 pts (CI -19.0 to -2.0, p=0.0195, n=147); A2: +20.4 pts (CI +12.2 to +28.6, p=5.3e-06, n=147)
- vs static_rm: A: -6.1 pts (CI -12.2 to +0.0, p=0.0931, n=147); B: +23.1 pts (CI +15.0 to +31.3, p=3.1e-07, n=147); A2: -2.7 pts (CI -11.6 to +6.1, p=0.652, n=147)
- vs static_rv: A: -4.8 pts (CI -10.9 to +1.4, p=0.21, n=147); B: +23.1 pts (CI +15.0 to +31.3, p=3.1e-07, n=147); A2: +10.9 pts (CI +4.1 to +17.7, p=0.00522, n=147)
- vs static_lww: A: -0.7 pts (CI -7.5 to +6.1, p=1, n=147); B: +23.1 pts (CI +15.0 to +31.3, p=1.41e-07, n=147); A2: +2.7 pts (CI -6.1 to +11.6, p=0.644, n=147)
- vs atmc_v6: A: +0.0 pts (CI -5.4 to +5.4, p=1, n=147); B: +7.5 pts (CI +0.0 to +15.0, p=0.0801, n=147); A2: +8.2 pts (CI +0.7 to +15.6, p=0.0576, n=147)

## Memory hygiene at the end of each phase (summed over seeds: audited / wrongly shown / wrongly hidden)
- static_rm: A: 88 / 3 / 0 | B: 158 / 88 / 0 | A2: 242 / 75 / 0
- static_rv: A: 89 / 4 / 0 | B: 76 / 4 / 0 | A2: 88 / 7 / 0
- static_lww: A: 84 / 4 / 0 | B: 113 / 45 / 0 | A2: 124 / 38 / 0
- atmc_v6: A: 78 / 0 / 1 | B: 161 / 3 / 0 | A2: 241 / 9 / 78
- rule_mem: A: 56 / 1 / 0 | B: 67 / 0 / 2 | A2: 61 / 0 / 1

Per-question agreement with static_rm in phase B: static_rv 139/147, static_lww 137/147

## Transfer stream (phase A: half 1 under the old rule; phase B: unseen functions under the new rule)
| condition | A | B | B vs no_memory |
|---|---|---|---|
| no_memory | 17.3 | 41.7 |  |
| static_rm | 34.7 | 26.4 | -15.3 pts (CI -29.2 to -1.4, p=0.0522, n=72) |
| static_rv | 30.7 | 29.2 | -12.5 pts (CI -26.4 to +1.4, p=0.122, n=72) |
| static_lww | 32.0 | 20.8 | -20.8 pts (CI -33.3 to -8.3, p=0.00592, n=72) |
| atmc_v6 | 25.3 | 30.6 | -11.1 pts (CI -22.2 to +0.0, p=0.0963, n=72) |
| rule_mem | 24.0 | 34.7 | -6.9 pts (CI -19.4 to +4.2, p=0.359, n=72) |
- B, rule_mem vs static_rm: +8.3 pts (CI -2.8 to +19.4, p=0.21, n=72)
- B, rule_mem vs static_rv: +5.6 pts (CI -4.2 to +15.3, p=0.424, n=72)
- B, rule_mem vs static_lww: +13.9 pts (CI +4.2 to +23.6, p=0.0213, n=72)
- B, rule_mem vs atmc_v6: +4.2 pts (CI -5.6 to +13.9, p=0.581, n=72)
