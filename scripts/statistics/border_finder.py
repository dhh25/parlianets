import pycountry
import pyogrio
import geopandas as gpd
import json

# 1) point to the Natural Earth countries file you downloaded
#    (choose either the “countries” or “tiny_countries” .shp, or concat both)
#shp_path = "~/parliamint/110m_cultural/ne_110m_admin_0_sovereignty.shp"
shp_path = "~/parliamint/110m_cultural/ne_110m_admin_0_countries.shp"
world = gpd.read_file(shp_path)

# 2) (optional) include the tiny‐countries layer if you need micro‐states
tiny = gpd.read_file("~/parliamint/110m_cultural/ne_110m_admin_0_tiny_countries.shp")
world = world._append(tiny, ignore_index=True)

iso3_to_iso2 = dict(zip(world["ISO_A3"], world["ISO_A2"]))

# 3) Index by ISO-A3 and build your ISO-A3 adjacency as before
world = world.set_index("ISO_A2_EH")

#world = world.set_index("ISO_A3")
sindex = world.sindex

neighbors_iso3 = {}
for iso3, row in world.iterrows():
    candidates = world.iloc[list(sindex.intersection(row.geometry.bounds))]
    touches = candidates[candidates.geometry.touches(row.geometry)].index.tolist()
    neighbors_iso3[iso3] = touches

# 4) Convert the keys and values from A3 → A2, dropping any unmapped codes
"""
neighbors_iso2 = {}
for iso3, neighs in neighbors_iso3.items():
    a2 = iso3_to_iso2.get(iso3)
    if not a2 or a2 == "":
        continue
    a2_list = [iso3_to_iso2.get(n) for n in neighs]
    # drop any None or empty strings
    neighbors_iso2[a2] = [x for x in a2_list if x]
"""

# 5) Dump to JSON
with open("neighbors_iso2.json", "w") as f:
    json.dump(neighbors_iso3, f, indent=2)

print("Wrote", len(neighbors_iso3), "alpha-2 codes with neighbors.")

"""
# 5) Dump to JSON
with open("neighbors_iso2.json", "w") as f:
    json.dump(neighbors_iso2, f, indent=2)

print("Wrote", len(neighbors_iso2), "alpha-2 codes with neighbors.")
"""


"""
# 3) make sure you have a simple country code column
#    (Natural Earth uses ISO_A3 for alpha-3 codes)
world = world.set_index("ISO_A3")

# 4) build a spatial index for performance
sindex = world.sindex

# 5) compute adjacency: “touches” means they share any boundary
neighbors = {}
for iso, row in world.iterrows():
    # find candidates whose bounding‐boxes intersect
    possible_idx = list(sindex.intersection(row.geometry.bounds))
    possible = world.iloc[possible_idx]
    # filter to those that actually touch
    neigh = possible[possible.geometry.touches(row.geometry)].index.tolist()
    neighbors[iso] = neigh

iso2neighbors = {}
for iso, neigh in neighbors.items():
    # convert to ISO_A2
    iso2neighbors[pycountry.countries.get(alpha_3=iso).alpha_2] = [
        pycountry.countries.get(alpha_3=n).alpha_2 for n in neigh
    ]

# 6) dump to JSON
with open("neighbors.json", "w") as f:
    json.dump(neighbors, f, indent=2)

"""