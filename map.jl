using XML
using PGFPlots

N = XML.read("DR_DEU_Roundabout_OF.osm_xy", Node) |>
    (m -> m.children) |> first |>
    (m -> filter(e -> e.tag == "node", m.children)) .|>
    (m -> (m["id"] => [parse(Float64, m["x"]), parse(Float64, m["y"])])) |>
    Dict

istype(way, type) = any(
    e.tag == "tag" && e["k"] == "type" && e["v"] == type for e in way.children)

M = XML.read("DR_DEU_Roundabout_OF.osm_xy", Node) |>
    (m -> m.children) |> first |>
    (m -> filter(e -> e.tag == "way", m.children)) |>
    (m -> filter(e -> istype(e, "curbstone"), m)) .|>
    (w -> filter(e -> e.tag == "nd", w.children)) .|>
    (w -> stack(N[n["ref"]] for n in w))

M = M .|>
    (w -> w .- [1000, 1000]) .|>
    (w -> w ./ 20)

save("map.pdf", Axis([
    Plots.Linear(m, style="black, no markers") for m in M
], xmin=-1, xmax=1, ymin=-1, ymax=1))
