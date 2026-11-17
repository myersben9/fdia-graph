"""The New York system: NYgrid's NPCC-140 case made solvable in AC, its reduced network, the 2019 loads by bus,
the hourly dispatch and the five-minute pool of AC operating states, and the timeline written from them.

It is one of the two grids a timeline is generated on, beside the IEEE cases: `source.NewYorkSource` is its
`generation.source.TimelineSource`: `source.generate(path)` writes its timeline by the path every timeline
takes (`NewYorkSource.production` builds the source of a release from a pool, `SingleEpisodeSource` the one of a
single episode), and `pool.PoolBuilder` builds the pool from the inputs `inputs.fetch_inputs` downloads. The
attack families it uses live in `fdia_graph.engine.attacks`, the meters in `fdia_graph.engine.sensing`, the
alarm rules in `fdia_analysis.se.detect` and the utility clients in `fdia_graph.generation.nygrid.ownership`. Import
each name from its module; importing this package alone loads neither pandapower nor Pyomo.
"""
