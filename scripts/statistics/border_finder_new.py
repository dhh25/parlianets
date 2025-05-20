import pycountry
import json

with open("scripts/statistics/restcountries_all.json", "r") as f:
    restcountries = json.load(f)

neighbors_iso2 = {}
for country in restcountries:
    neighbors_iso2[country["cca2"]] = []
    if "borders" not in country:
        continue
    else:
        for neighbor in country["borders"]:
            if neighbor == "UNK": #exception for kosovo (UNK), because UNK is not known in pycountry but present in the data
                neighbor_str = "XK"
            else:
                neighbor_obj = pycountry.countries.get(alpha_3=neighbor)
                neighbor_str = neighbor_obj.alpha_2

            neighbors_iso2[country["cca2"]].append(neighbor_str)


with open("neighbors_iso2.json", "w") as f:
    json.dump(neighbors_iso2, f, indent=2)

print("Wrote", len(neighbors_iso2), "alpha-2 codes with neighbors.")