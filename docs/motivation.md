## Motivation

Training deep learning models, data loading can quickly become a bottleneck. When we cannot preload the dataset into memory, we have to load and preprocess it batch by batch during training.

To not slow down the training, loading a batch must

- not take longer than one training step
- not block the main thread; and
- avoid garbage collection pauses

`DataLoaders`

- uses multiple worker threads to maximize throughput
- keeps the main thread free for the training step; and
- allows buffered data loading for supporting data containers to reduce allocations


