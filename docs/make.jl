using Publish
using DataLoaders

p = Publish.Project(DataLoaders)

# needed to prevent error when overwriting
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)

# build documentation
deploy(DataLoaders; root = "/DataLoaders.jl", force = true, label = "dev")
