from datascience.tools.ign.ign_sparse_raster import create_ign_sparse, extract_from_ign_sparse

create_ign_sparse(source_occ='glc19_pl_complete', source_ign='ign_5m_maps_and_patches', patch_size=51)

# extract_from_ign_sparse(rasters=('red.npz', 'green.npz', 'blue.npz'), pos=(43.577232, 3.937999), size=51, res=5.0, top_left=(-360000, 7240000))
