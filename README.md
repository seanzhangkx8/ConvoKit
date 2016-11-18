# Installing
Run `python3 setup.py install` to install the package.
Use `import socialkit` to import it into your project.

# Basic usage
1. Load model: `model = Model(filename=...)`
2. Create coordination object: `coord = Coord(model)`
3. Define groups using `model.users` or `model.user_names`:
    - Option 1. Merge users with the same name (for example to group justices across cases):
            `group_A = model.user_names(lambda user: user.info["is-justice"])  # ["justice roberts", ...]`
    - Option 2. By user "object" (eg if we want to have a separate justice user for each case):
            `group_A = model.users(lambda user: user.info["is-justice"])  # [roberts-case1, ginsburg-case3, ...]`
4. Compute coordination: `scores = model.score(group_A, group_B)`
5. (Optional) get aggregate scores:
        `average_by_marker_agg1, average_by_marker, agg1, agg2, agg3 = model.score_report(scores)`
