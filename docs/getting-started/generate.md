# Generate your first timeline

This tutorial generates a small timeline on IEEE-14 and loads it. It takes about five minutes on
a laptop.

## Install

```bash
pip install "fdia-graph[generate,torch]"
```

The `generate` extra adds pandapower, which the generator uses to read the IEEE cases.

## 1. Generate

```python
import fdia_graph as fg

if __name__ == "__main__":
    path = fg.generate("ieee14", name="first_run", frames=2000)
    print(path)
```

What this does:

1. **Downloads the pool.** The IEEE-14 pool of operating points is a file of about 25 MB,
   downloaded once and kept in `~/.cache/fdia_graph`. A pool is the true state of the grid at
   every frame.
2. **Walks 2,000 frames.** It places attack episodes, designs each one, and writes every frame
   with meter noise.
3. **Writes one file** and registers it under the name `first_run`.

Most of the time is the design of the overload attacks
([measured per episode](../concepts/attacks.md#the-search-based-designs)).

## 2. Load it

```python
train = fg.load("first_run", split="train")
print(train.summary())
```

```
Summary(system=14, N=14, E=20, n=1200,
        families={'benign': 600, 'At': 300, 'Am': 300})
```

The train split is the first 60 % of the frames. Half of them are under attack, shared equally
between the two families the generator makes.

| Family | What the attacker does |
|---|---|
| `At` | a false load that rises a little each frame |
| `Am` | the reported flow of two lines rises to their ratings |

```python
episodes = train.episodes
print(episodes.length[:5])  # [60 60 60 60 60]
print(episodes.family[:5])  # codes: 5 is At, 7 is Am
```

Each episode is 60 frames. From here the file loads like a released one
([Load a dataset](load.md)).

## 3. Change a setting

The common settings are keywords of `fg.generate`. The others are fields of the settings
models it takes.

```python
if __name__ == "__main__":
    fg.generate(
        "ieee14",
        name="ramps_only",
        frames=2000,
        families=("At",),
        attacked_frac=0.3,
        ramp_len=30,
        seed=7,
    )
```

This one places only slow ramps, 30 frames each, on 30 % of the frames. With no overload attack
to design it runs much faster. [The settings tables](../guides/generation.md#the-settings) list
every setting.

A timeline is fixed by its settings, its seed and its pool: the same call writes the same file.

## The same on the New York case

The New York case goes through the same generator, with the same settings and other data.
Nothing is released for it, so you build its pool first. That needs a checkout of the
repository, the `[ny]` extra and the Ipopt solver, and several hours of solving.
[Build the operating points](../guides/operating_points.md#the-new-york-case-build-the-pool)
has the steps, and
[Generate a timeline](../guides/generation.md#the-original-generator-on-the-new-york-case)
the build of the timeline from the pool.

## Where next

- [Analyse a timeline](analyse.md): run an estimator and a localizer on the file.
- [Generate a timeline](../guides/generation.md): the steps, and every setting.
- [Follow one attack through the code](../guides/tour.md): one episode, class by class.
- [The attack families](../concepts/attacks.md): what the two families are.
