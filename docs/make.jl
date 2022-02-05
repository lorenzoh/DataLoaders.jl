using Pkg
using Pollen

using DataLoaders
const PACKAGE = DataLoaders


# Create target folder
DIR = abspath(mkpath(ARGS[1]))


# Create Project
m = PACKAGE
ms = [PACKAGE,]


@info "Creating project..."
project = Project(
    Pollen.Rewriter[
        Pollen.DocumentFolder(pkgdir(m), prefix = "documents"),
        Pollen.ParseCode(),
        Pollen.ExecuteCode(),
        Pollen.PackageDocumentation(ms),
        Pollen.DocumentGraph(),
        Pollen.SearchIndex(),
    ],
)

@info "Rewriting documents..."
Pollen.rewritesources!(project)

@info "Writing to disk at \"$DIR\"..."
builder = Pollen.FileBuilder(
    Pollen.JSON(),
    DIR,
)
Pollen.build(
    builder,
    project,
)
