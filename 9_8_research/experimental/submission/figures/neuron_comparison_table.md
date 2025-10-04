
# Neuron Firing Comparison: Buggy vs Non-Buggy Responses

| Layer | Buggy Response (Chat Template) | Non-Buggy Response (Simple Format) |
|-------|-------------------------------|-----------------------------------|
| **7** | Neuron 1978 | - |
| **11** | - | Neuron 11862 |
| **13** | Neuron 10352 | Neuron 10352 ✓ |
| **14** | Neuron 13315 (2x), 2451, **12639** ⚠️ | Neuron 13315 (2x), **12639** ⚠️ |
| **15** | Neuron 3136 (2x), 5076, 421 | Neuron 3136 (2x), 5076 (2x), 421 |
| **28** | Neuron 10823, 8818, 5336 | Neuron 11450, 12900, 10823 |
| **29** | Neuron 664, 12248, 1435 | Neuron 12248, 10726, 2836 |
| **30** | Neuron 840, 13679, 7305 | Neuron 840, 14215 (multiple) |
| **31** | Neuron 801, 4581, 13336 (12.0), 12004 (11.5), 9692, 2398, 12111 | Neuron 801, 4581, 13336 (14.8) ↑, 12004 (14.4) ↑, 9692, 2398 |

**Key:**
- ⚠️ = Entangled neuron (fires in both buggy and correct responses)
- ✓ = Shared neuron
- ↑ = Higher activation in correct response
- Layers 2-15: Hijacker Circuit
- Layers 28-31: Reasoning Circuit

**Critical Finding:** Layer 14, Neuron 12639 is active in BOTH responses, demonstrating irremediable entanglement.
