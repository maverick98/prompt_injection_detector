# Game-Theoretic Attacker/Defender Analysis

This analysis treats prompt-injection defense as a finite zero-sum game.

- Attacker actions: red-team evasion strategies.
- Defender actions: detector threshold policies.
- Defender loss: bypass rate plus `1.000` times false-positive rate.
- Defender objective: minimize worst-case security and usability loss.

## Equilibrium

Solver: `scipy_linprog_highs`

Game value, interpreted as expected defender loss under equilibrium play:
`0.100`

## Attacker Mixed Strategy

- `paraphrase`: 0.000
- `obfuscation`: 0.000
- `roleplay_wrapping`: 0.000
- `encoding`: 0.000
- `nested_injection`: 1.000

## Defender Mixed Strategy

- threshold `0.250`: 0.000
- threshold `0.410`: 0.000
- threshold `0.475`: 1.000
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
| paraphrase | 0.125 | 0.0875 | 0.075 | 0.05 | 0.075 |
| obfuscation | 0.125 | 0.1 | 0.0875 | 0.0625 | 0.0875 |
| roleplay_wrapping | 0.125 | 0.0875 | 0.075 | 0.05 | 0.05 |
| encoding | 0.125 | 0.0875 | 0.075 | 0.0375 | 0.025 |
| nested_injection | 0.125 | 0.1 | 0.1 | 0.1 | 0.25 |
