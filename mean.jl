using CSV
using DataFrames
using DynamicPolynomials
using LinearAlgebra
using MomentOpt
using MosekTools
using PGFPlots
using XML

istype(way, type) = any(
    e.tag == "tag" && e["k"] == "type" && e["v"] == type for e in way.children)

map_node = XML.read("DR_DEU_Roundabout_OF.osm_xy", Node) |>
    (m -> m.children) |> first |>
    (m -> filter(e -> e.tag == "node", m.children)) .|>
    (m -> (m["id"] => [parse(Float64, m["x"]), parse(Float64, m["y"])])) |>
    Dict

map_ways = XML.read("DR_DEU_Roundabout_OF.osm_xy", Node) |>
    (m -> m.children) |> first |>
    (m -> filter(e -> e.tag == "way", m.children)) |>
    (m -> filter(e -> istype(e, "curbstone"), m)) .|>
    (w -> filter(e -> e.tag == "nd", w.children)) .|>
    (w -> stack(map_node[n["ref"]] for n in w)) .|>
    (w -> w .- [1000, 1000]) .|>
    (w -> w ./ 20)

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","track_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 0 1000 1000 0 0]) |>
    (d -> d ./ [1 1 20 20 20 20]) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

@polyvar t x[1:2] u[1:2]
frames = unique(D[:, "frame_id"])
σ = D[:,["x","y","vx","vy"]] |> eachrow .|> collect .|> 
    (s -> DiracMeasure([x;u],s)) |> sum |> 
    (m -> m * (1/length(frames)))

@polyvar x1[1:2] u1[1:2] x2[1:2] u2[1:2]
d = 3
k = let v1 = monomials([x1;u1],0:d);
    v2 = monomials([x2;u2],0:d);
    v = monomials([x;u],0:d);
    W = integrate.(v*v',[σ]);
    v1'*inv(W+1e-4I)*v2
end
k11 = subs(k, ([x1;u1;x2;u2] .=> [x1;u1;x1;u1])...)
k12 = subs(k, ([x1;u1;x2;u2] .=> [x1;u1;x2;u2])...)
k21 = subs(k, ([x1;u1;x2;u2] .=> [x2;u2;x1;u1])...)
k22 = subs(k, ([x1;u1;x2;u2] .=> [x2;u2;x2;u2])...)
l = k11 + k12 + k21 + k22

F0 = Int.(frames[1:10:100])

for f0 in F0

    X0 = filter(e -> e["frame_id"] == f0, D)
    N = size(X0,1)

    p = GMPModel(Mosek.Optimizer)
    @variable p ρ[1:N]  Meas([t;x1;u1;x2;u2], support=@set([t;x1;u1;x2;u2]'*[t;x1;u1;x2;u2] <= 10))
    @variable p ρ0[1:N] Meas([t;x1;u1;x2;u2], support=@set([t;x1;u1;x2;u2]'*[t;x1;u1;x2;u2] <= 10 && t == 0))
    @variable p ρT[1:N] Meas([t;x1;u1;x2;u2], support=@set([t;x1;u1;x2;u2]'*[t;x1;u1;x2;u2] <= 10 && t == 3))

    ϕ = monomials([t;x1;x2], 0:2d)
    @constraint p [i=1:N] Mom.(differentiate.(ϕ,t) + differentiate(ϕ,[x1;x2])*[u1;u2], ρ[i]) .== Mom.(ϕ, ρT[i] - ρ0[i])
    @constraint p [i=1:N] Mom.(ϕ,ρ0[i]) .== integrate.(ϕ, sum(DiracMeasure(
        [t;x1;u1;x2;u2], [0;collect(X0[i,["x","y","vx","vy"]]);collect(X0[j,["x","y","vx","vy"]])]
    ) for j=1:N if j != i))
    let v1 = monomials([t;x1;u1], 0:2d);
        v2 = monomials([t;x2;u2], 0:2d);
        @constraint p [i=1:N] Mom.(v2, ρ[i]) .== Mom.(v1, sum(ρ[j] for j=1:N if j != i)/(N-1))
    end

    @objective p Min Mom(l,sum(ρ)/N^2)
    optimize!(p)

    q = [let v = monomials(x1, 0:d);
        W = integrate.(v*v', [m]);
        v'*inv(W + 1e-4I)*v
    end for m in ρ]

    μT = integrate.(x1,ρT') / (N-1)
    fT = f0 + 3*10
    XT = filter(e -> e["frame_id"] == fT && e["track_id"] in collect(X0[:,"track_id"]), D)

    save("mean-f$(f0).pdf", Axis([
        Plots.Image((x,y) -> sum(1/qi(x,y) for qi in q), (-1,1), (-1,1));
        Plots.Quiver(
            D[1:50:end,"x"],    D[1:50:end,"y"],
            D[1:50:end,"vx"]/3, D[1:50:end,"vy"]/3,
            style="-stealth, no markers, blue",
        );
        Plots.Scatter(X0[:,"x"], X0[:,"y"]);
        Plots.Scatter(μT, style="mark=x, red");
        Plots.Scatter(XT[:,"x"], XT[:,"y"], style="mark=x, green");
        [Plots.Linear(m, style="white, no markers") for m in map_ways]
    ], xmin=-1, xmax=1, ymin=-1, ymax=1))

end
