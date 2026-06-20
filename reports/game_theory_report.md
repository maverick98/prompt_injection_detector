# Game-Theoretic Attacker/Defender Analysis

This analysis treats prompt-injection defense as a finite zero-sum game.

- Attacker actions: red-team evasion strategies.
- Defender actions: detector threshold policies.
- Defender loss: bypass rate plus `0.250` times false-positive rate.
- Defender objective: minimize worst-case security and usability loss.

## Equilibrium

Solver: `scipy_linprog_highs`

Game value, interpreted as expected defender loss under equilibrium play:
`0.031`

## Attacker Mixed Strategy

- `paraphrase`: 0.000
- `obfuscation`: 1.000
- `roleplay_wrapping`: 0.000
- `encoding`: 0.000
- `nested_injection`: 0.000

## Defender Mixed Strategy

- threshold `0.250`: 1.000
- threshold `0.410`: 0.000
- threshold `0.475`: 0.000
- threshold `0.560`: 0.000
- threshold `0.700`: 0.000

## False-Positive Rates By Threshold

- threshold `0.250`: 0.125
- threshold `0.410`: 0.087
- threshold `0.475`: 0.075
- threshold `0.560`: 0.037
- threshold `0.700`: 0.025

## Loss Matrix

Rows are attacker strategies. Columns are defender thresholds. Values are bypass
rate plus weighted false-positive burden, so lower is better for the defender.

| attacker_strategy | 0.25 | 0.41 | 0.475 | 0.56 | 0.7 |
| --- | --- | --- | --- | --- | --- |
| paraphrase | 0.03125 | 0.021875 | 0.01875 | 0.021875 | 0.05625 |
| obfuscation | 0.03125 | 0.034375 | 0.03125 | 0.034375 | 0.06875 |
| roleplay_wrapping | 0.03125 | 0.021875 | 0.01875 | 0.021875 | 0.03125 |
| encoding | 0.03125 | 0.021875 | 0.01875 | 0.009375 | 0.00625 |
| nested_injection | 0.03125 | 0.034375 | 0.04375 | 0.071875 | 0.23125 |
