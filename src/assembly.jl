# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Base.Threads

function isapprox(a1::Assembly, a2::Assembly)
    T = isapprox(a1.K, a2.K)
    T &= isapprox(a1.C1, a2.C1)
    T &= isapprox(a1.C2, a2.C2)
    T &= isapprox(a1.D, a2.D)
    T &= isapprox(a1.f, a2.f)
    T &= isapprox(a1.g, a2.g)
    return T
end

function assemble_prehook!
end

function assemble_posthook!
end

""" Assemble problem, i.e. construct global stiffness matrix and force vector
for field problem, constraint matrices C1, C2, D and g for boundary problems
for all for mixed problems.

Parameters
----------

auto_initialize
    initialize problem before assembling automatically (solver will do this anyway)
threded
    use multithreading when assembling
parallel
    use distributed computing when assembling

"""
function assemble!(problem::Problem, time::Real; auto_initialize=true, threaded=false, parallel=false)
    if !isempty(problem.assembly)
        warn("Assemble problem $(problem.name): problem.assembly is not empty and assembling, are you sure you know what are you doing?")
    end
    if isempty(problem.elements)
        warn("Assemble problem $(problem.name): problem.elements is empty, no elements in problem?")
    else
        first_element = first(problem.elements)
        unknown_field_name = get_unknown_field_name(problem)
        if !haskey(first_element, unknown_field_name)
            warn("Assemble problem $(problem.name): seems that problem is uninitialized.")
            if auto_initialize
                info("Initializing problem $(problem.name) at time $time automatically.")
                initialize!(problem, time)
            end
        end
    end
    if method_exists(assemble_prehook!, Tuple{typeof(problem), Float64})
        assemble_prehook!(problem, time)
    end

    chunks(a, n) = [a[i:n:end] for i=1:n]

    function assemble_elements(elements)
        assembly = Assembly()
        for element in elements
            assemble!(assembly, problem, element, time)
        end
        return assembly
    end

    if threaded && !parallel
        n_chunks = nthreads()
        info("Using threading ($(nthreads()) threads) in assemble, splitting problem to $n_chunks chunks.")
        element_subsets = chunks(get_elements(problem), n_chunks)
        assemblies = [Assembly() for i=1:n_chunks]
        @threads for i=1:n_chunks
            for element in element_subsets[i]
                assemble!(assemblies[i], problem, element, time)
            end
        end
        for subassembly in assemblies
            append!(problem.assembly, subassembly)
        end
    elseif !threaded && parallel
        n_chunks = nworkers()
        info("Using parallel map ($(nworkers()) workers) in assemble, splitting problem to $n_chunks chunks.")
        element_subsets = chunks(get_elements(problem), n_chunks)
        assemblies = pmap(assemble_elements, element_subsets)
        for subassembly in assemblies
            append!(problem.assembly, subassembly)
        end
    else
        for element in get_elements(problem)
            assemble!(problem.assembly, problem, element, time)
        end
    end


    if method_exists(assemble_posthook!, Tuple{typeof(problem), Float64})
        assemble_posthook!(problem, time)
    end

    return true
end

function assemble!(problem::Problem, time::Real, ::Type{Val{:mass_matrix}}; density=0.0, dual_basis=false, dim=0)
    if !isempty(problem.assembly.M)
        warn("problem.assembly.M is not empty and assembling, are you sure you know what are you doing?")
    end
    if dim == 0
        dim = get_unknown_field_dimension(problem)
    end
    for element in get_elements(problem)
        if !haskey(element, "density") && density == 0.0
            error("Failed to assemble mass matrix, density not defined!")
        end
        nnodes = length(element)
        M = zeros(nnodes, nnodes)
        for ip in get_integration_points(element, 1)
            detJ = element(ip, time, Val{:detJ})
            N = element(ip, time)
            rho = haskey(element, "density") ? element("density", ip, time) : density
            M += ip.weight*rho*N'*N*detJ
        end
        gdofs = get_gdofs(problem, element)
        for j=1:dim
            ldofs = gdofs[j:dim:end]
            add!(problem.assembly.M, ldofs, ldofs, M)
        end
    end
end

