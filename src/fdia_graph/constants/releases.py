"""The data releases: where the built-in datasets are published, which release is pinned, and the
sha256 of every asset of the releases the registry verifies."""

from __future__ import annotations

from .grid import IEEE_CASES

REPO = "myersben9/fdia-graph"  # the GitHub repository whose releases carry the data assets
PINNED_RELEASE = "v0.8.3"  # the data release `load(release=None)` reads (FDIA_GRAPH_RELEASE overrides)
FIRST_RELEASE = (0, 8, 0)  # the first data release of timelines, the only layout the package reads
DATA_TAG_PREFIX = "data-"  # GitHub tags of data releases from v0.8.0: "data-v0.8.0"

# Canonical name -> bus count: every entry is downloadable at every data release.
BUILTIN = {f"ieee{C}": {"system": C} for C in IEEE_CASES}
# A bus count (as text or as a number) -> the canonical "ieeeNN" name.
ALIASES = {key: f"ieee{C}" for C in IEEE_CASES for key in (str(C), C)}
# sha256 of the timeline assets, per release; a release not listed downloads unverified.
SHA256: dict[str, dict[str, str]] = {
    "v0.8.0": {
        "ieee14": "c1d3fafcb37e3c2b57a07523fee6bd0a658ed9cd4b77bc6b38697762e13a5a28",
        "ieee30": "9ad2bd82adef5dbd8c08587982c0f50bafe584c8318f13e4b8082795f7418a66",
        "ieee57": "085a5fbf6c0b743f04ba8d290dcd09fb6151e8bf0173250258ef3565cfca8948",
        "ieee89": "47fbf40c061852fe0c39a43105036bd27c2f2b34c8cb52dc31a43204acd91789",
        "ieee118": "bb5fbda40e6643d47938b23b63704ab8bb81d07383678fa394b7efe159cf2dd5",
        "ieee145": "aaa5fc23f93d138cbebb25724d5479b532fab7fdffa2b55738c6018aa936f9e7",
        "ieee200": "be4ad6a6d598aac9b4fe31c05048c50f50086822140c2001ce0ecdad77edb697",
        "ieee300": "b4f08db8b643a58d7050e2079b5890c3acf4f09712ab6efa6cb7f8f165232e7f",
    },
    "v0.8.1": {
        "ieee14": "6ed9df442a1faee8271267034485c065c15c6b6b099db476e8fad8633cd22198",
        "ieee30": "fcc50149715f9b8965d503422f7bd6f8308a3487f28e2b8c03fedf3c956c6887",
        "ieee57": "de52d152fcf207f197755044485b4c6209617e2ec5ff03f6c3551f03c15a06bd",
        "ieee89": "74b8512b1274123a11db6b312d4bc197f23c2f5c44770f0f20b6f7abd856e7e0",
        "ieee118": "cfa713d785b3db34f2edba6ec6d78927c97e14cafd3ce2106a1cc5e1cc89d4f7",
        "ieee145": "2471510912ae82cb27743ded25f2bf9c562e6f58f6a0de0ad3afa0e060e9008c",
        "ieee200": "be4ad6a6d598aac9b4fe31c05048c50f50086822140c2001ce0ecdad77edb697",
        "ieee300": "67b5d727ed498bba270dcc5eb230afec287c4f1fd55b74178a453985edbcd75f",
    },
    "v0.8.3": {
        "ieee14": "11460614f78eb0770dc47adcf53bca20f8692b348e2d7cc048e6f714bd40a7b3",
        "ieee30": "87312be35a74ea341cacd4f4cf306af04e875f90f50ac472eccc715213de31b3",
        "ieee57": "05522321703ab0fba3755b9d8582a7a420abe0c5ad24b535667201cd4c5666f6",
        "ieee89": "206adfed97c37d6bf38990d8abfac44154f4fa461be3ddeea74b9f1e4c09fd8d",
        "ieee118": "dfc47298b77daca47e2e581d2a60163b094d6a77411604605d8f1d4b618f71ae",
        "ieee145": "5d3d0b6febf4545c3f73f749a56368ade9b64101113252e302448b7d00fd24b3",
        "ieee200": "9afe6f20dc327fb700303420e8f5da04b998bf2f74956d6982fb8ab845545f47",
        "ieee300": "a6948a4f0649cf3823bbea806d3084539f8088c207746c782fce58577dc59c14",
    },
}
