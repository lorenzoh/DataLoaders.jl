# Ecosystem

{.subtitle}
Overview of packages that DataLoaders.jl builds on or that use it.

This package is part of an ecosystem of packages providing useful tools for machine learning in Julia. These compose nicely due to shared interface packages like [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) and the natural extensibility that Julia's multiple dispatch provides. DataLoaders.jl works with any package the implements the [data container interface](interface.md). This means you can easily drop it in to an existing workflow or use the functionality of other packages to work with DataLoaders.jl more effectively.

The most important package for manipulating data containers is [**MLDataPattern.jl**](https://github.com/JuliaML/MLDataPattern.jl) which provides a large set of tools for transforming and composing data containers. Some examples are given here: [Shuffling, subsetting, splitting](shuffling.md)

[**MLDatasets.jl**](https://github.com/JuliaML/MLDatasets.jl) makes it easy to load common benchmark datasets as data containers.

A package that makes heavy use of DataLoaders.jl to train large deep learning models is [**FastAI.jl**](https://github.com/FluxML/FastAI.jl). It also provides many easy-to-load data containers for larger computer vision, tabular, and NLP datasets.