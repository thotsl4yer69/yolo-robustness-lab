# Architecture

```text
dataset -> sample discovery -> ground truth
                    |
                    +-> transformation suite -> detector backend
                                                   |
                                                   v
                                               predictions
                                                   |
                                                   v
                                               IoU matcher
                                                   |
                                                   v
                                                 metrics
                                                /       \
                                               v         v
                                            CSV/JSON   annotations
                                                         |
                                                         v
                                                       plots
```

The detector backend is isolated so future detector implementations can be added without changing benchmark logic. Transformations are deterministic when seeded. Every run stores configuration and raw predictions for auditability.
