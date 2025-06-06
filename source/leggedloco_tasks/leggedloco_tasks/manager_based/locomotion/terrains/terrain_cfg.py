import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

##
# Configuration for sub terrains.
##

FLAT = terrain_gen.MeshPlaneTerrainCfg(
    proportion=0.4
)

RANDOM_ROUGH = terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.4, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
)

HF_PYRAMID_SLOPE = terrain_gen.HfPyramidSlopedTerrainCfg(
    proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
)

HF_PYRAMID_SLOPE_INV = terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
    proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
)

BOX = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.1), platform_width=2.0
)

GAP = terrain_gen.MeshGapTerrainCfg(
    proportion=0.2, gap_width_range=(0.5, 1.0), platform_width=2.0
)

PYRAMID_STAIRS = terrain_gen.MeshPyramidStairsTerrainCfg(
    proportion=0.2,
    step_height_range=(0.06, 0.20),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    holes=False,
)

PYRAMID_STAIRS_INV = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
    proportion=0.2,
    step_height_range=(0.06, 0.20),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    holes=False,
)



##
# Configuration for custom terrains.
##

FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": FLAT.replace(proportion=0.5),
        "random_rough": RANDOM_ROUGH.replace(proportion=0.5),
    },
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": PYRAMID_STAIRS.replace(proportion=0.3),
        "pyramid_stairs_inv": PYRAMID_STAIRS_INV.replace(proportion=0.4),
        "random_rough": RANDOM_ROUGH.replace(proportion=0.1),
        "hf_pyramid_slope": HF_PYRAMID_SLOPE.replace(proportion=0.1),
        "hf_pyramid_slope_inv": HF_PYRAMID_SLOPE_INV.replace(proportion=0.1),
    },
)