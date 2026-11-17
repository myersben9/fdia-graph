"""The toolkit's constants, one module per subject: facts only, each with its unit and its source.

A constant is a value nothing in a run changes: a physical or regulatory fact, a column position, a
code a file carries, a pinned input, a number fitted once. A value a user tunes is a knob and is a
field of a settings model (`models.config`); the names of a timeline file's groups, datasets and
attributes are in `schema.py`. Every name is imported from its module (`constants.channels`,
`constants.meters`, ...); this package exports none, and its modules import nothing from the rest
of the toolkit, so any module may import them.
"""
