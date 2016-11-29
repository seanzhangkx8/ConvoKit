# Cornell Conversational Analysis Toolkit
## Installing
Run `python3 setup.py install` to install the package.
Use `import convokit` to import it into your project.

## Examples
See `examples/coordination/` for examples on computing coordination (see the [Echoes of Power](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) paper)

## Basic usage
1. Load corpus: `corpus = Corpus(filename=...)`
2. Create coordination object: `coord = Coord(corpus)`
3. Define groups using `corpus.users`:
        `group_A = corpus.users(lambda user: user.info["is-justice"])  # [roberts, ginsburg, ...]`
4. Compute coordination: `scores = coord.score(group_A, group_B)`
5. (Optional) get aggregate scores:
        `average_by_marker_agg1, average_by_marker, agg1, agg2, agg3 = coord.score_report(scores)`
