"""The New York system: NYgrid's NPCC-140 case made solvable in AC, its reduced network, the 2019 loads by bus,
the hourly dispatch and the five-minute pool of AC operating states, and the timeline written from them.

`inputs.fetch_inputs` downloads the inputs, `pool.PoolBuilder` builds the pool, and `source.NewYorkSource`
(a `generation.source.TimelineSource`) writes the timeline the way every grid does. The attack families
live in `fdia_graph.engine.attacks`, the meters in `fdia_graph.engine.sensing`, the utility clients in
`fdia_graph.generation.nygrid.ownership`. Import each name from its module; importing this package alone
loads neither pandapower nor Pyomo.
"""
