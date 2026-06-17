| Métrique | Baseline (CPU) | Neo (zero-copy) | Gain |
|---|---|---|---|
| **Mono-flux** end-to-end | 43.7 FPS | 58.0 FPS | ×1.33 |
| CPU (process) | 87.3% | 15.0% | ÷5.8 |
| Pixels sur PCIe / frame | 4.92 MB | 0 MB | zero-copy |
| decode / frame | 3.34 ms | 2.07 ms |  |
| préprocessing / frame | 3.38 ms | 0.22 ms |  |
| upload PCIe / frame | 0.74 ms | 0 ms |  |
| inférence / frame | 15.36 ms | 14.89 ms |  |
| **Multi-flux 3x** cumulé | 83.6 FPS | 107.4 FPS | ×1.28 |
| CPU (process, 3x) | 86.9% | 11.1% | ÷7.8 |
| **Multi-flux 6x** cumulé | 99.7 FPS | 142.2 FPS | ×1.43 |
| CPU (process, 6x) | 93.9% | 9.4% | ÷10.0 |
