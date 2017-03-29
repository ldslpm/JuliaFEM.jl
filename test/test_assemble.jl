# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM

function get_problem(N)
    problem = Problem(Elasticity, "tet10", 3)
    for i=1:N
        element = Element(Tet10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        update!(element, "youngs modulus", 480.0)
        update!(element, "poissons ratio", 1/3)
        x1 = [2.0, 3.0, 4.0]
        x2 = [6.0, 3.0, 2.0]
        x3 = [2.0, 5.0, 1.0]
        x4 = [4.0, 3.0, 6.0]
        x5 = 0.5*(x1+x2)
        x6 = 0.5*(x2+x3)
        x7 = 0.5*(x3+x1)
        x8 = 0.5*(x1+x4)
        x9 = 0.5*(x2+x4)
        x10 = 0.5*(x3+x4)
        X = Dict{Int64, Vector{Float64}}(
            1 => x1, 2 => x2, 3 => x3, 4 => x4, 5 => x5,
            6 => x6, 7 => x7, 8 => x8, 9 => x9, 10 => x10)
        u = Dict{Int64, Vector{Float64}}()
        for i=1:10
            u[i] = [0.0, 0.0, 0.0]
        end
        update!(element, "geometry", X)
        update!(element, "displacement", 0.0 => u)
        ips = get_integration_points(element)
        push!(problem.elements, element)
    end
    initialize!(problem, 0.0)
    assemble!(problem, 0.0)
    empty!(problem.assembly)
    return problem
end

for N in [1, 10, 100, 1000]
    info("$N elements")
    problem = get_problem(N)
    @time assemble!(problem, 0.0)
end
