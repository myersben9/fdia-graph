"""The New York system: NYgrid's NPCC-140 case made solvable in AC, its reduced network, the 2019 loads by bus,
the hourly dispatch and the five-minute pool of AC operating states, and the timeline written from them.

`inputs.fetch_inputs` downloads the inputs, `pool.PoolBuilder` builds the pool, and `source.NewYorkSource`
(a `generation.source.TimelineSource`) writes the timeline the way every grid does. The grid itself (`Case`,
`Network`, the case file `case_nygrid.m`, the utility that owns each bus) is in `fdia_graph.systems`. The
attack families live in `fdia_graph.engine.attacks`, the meters in `fdia_graph.sensing`, the utility clients in
`fdia_graph.systems.ownership`. Import each name from its module; importing this package alone
loads neither pandapower nor Pyomo.
"""
