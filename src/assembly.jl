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

function assemble!(assembly::Assembly, problem::Problem, elements::Vector{Element}, time::Real)
    warn("assemble!() this is default assemble operation, decreased performance can be expected without preallocation of memory!")
    for element in elements
        assemble!(assembly, problem, element)
    end
    return nothing
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

    if threaded && !parallel
        n_chunks = nthreads()
        info("Using threading ($(nthreads()) threads) in assemble, splitting problem to $n_chunks chunks.")
        sub_elements = chunks(get_elements(problem), n_chunks)
        sub_assemblies = [Assembly() for i=1:n_chunks]
        sub_problems = [copy(problem) for i=1:n_chunks]
        @threads for i=1:n_chunks
            assemble!(sub_assemblies[i], sub_problems, sub_elements[i], time)
        end
        for sub_assembly in sub_assemblies
            append!(problem.assembly, sub_assembly)
        end
    elseif !threaded && parallel
        n_chunks = nworkers()
        info("Using parallel map ($(nworkers()) workers) in assemble, splitting problem to $n_chunks chunks.")
        sub_elements = chunks(get_elements(problem), n_chunks)
        sub_problems = [copy(problem) for i=1:n_chunks]
        input_data = collect(zip(sub_problems, sub_elements))

        function parallel_assemble(input_data)
            assembly = Assembly()
            problem, elements = input_data
            assemble!(assembly, problem, elements, time)
            return assembly
        end

        sub_assemblies = pmap(parallel_assemble, input_data)

        for sub_assembly in sub_assemblies
            append!(problem.assembly, sub_assembly)
        end
    else
        assemble!(problem.assembly, problem, problem.elements, time)
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

