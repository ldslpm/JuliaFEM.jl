# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractElement

type Element{E<:AbstractElement}
    id :: Int
    connectivity :: Vector{Int}
    integration_points :: Vector{IP}
    fields :: Dict{String, Field}
    properties :: E
end

""" Construct a new element of type E.

Examples
--------
julia> element = Element(Tri3, [1, 2, 3])
"""
function Element{E<:AbstractElement}(::Type{E}, connectivity::Vector{Int})
    return Element{E}(-1, connectivity, [], Dict(), E())
end

function getindex(element::Element, field_name::AbstractString)
    return element.fields[field_name]
end

function setindex!(element::Element, data::Field, field_name)
    element.fields[field_name] = data
end

function get_element_type{E}(element::Element{E})
    return E
end

function get_element_id{E}(element::Element{E})
    return element.id
end

function is_element_type{E}(element::Element{E}, element_type)
    return is(E, element_type)
end

function filter_by_element_type(element_type, elements)
    return filter(element -> is_element_type(element, element_type), elements)
end

function setindex!(element::Element, data::Function, field_name)
    if method_exists(data, Tuple{Element, Vector, Float64})
        # create enclosure to pass element as argument
        function wrapper_(ip, time)
            return data(element, ip, time)
        end
        field = Field(wrapper_)
    else
        field = Field(data)
    end
    element.fields[field_name] = field
end

function setindex!(element::Element, data, field_name)
    element.fields[field_name] = Field(data)
end

""" Return a Field object from element.

Examples
--------
>>> element = Element(Seg2, [1, 2])
>>> data = Dict(1 => 1.0, 2 => 2.0)
>>> update!(element, "my field", data)
>>> element("my field")

"""
function (element::Element)(field_name::String)
    return element[field_name]
end

""" Return a Field object from element and interpolate in time direction.

Examples
--------
>>> element = Element(Seg2, [1, 2])
>>> data1 = Dict(1 => 1.0, 2 => 2.0)
>>> data2 = Dict(1 => 2.0, 2 => 3.0)
>>> update!(element, "my field", 0.0 => data1, 1.0 => data2)
>>> element("my field", 0.5)

"""
function (element::Element)(field_name::String, time::Float64)
    return element[field_name](time)
end

function last(element::Element, field_name::String)
    return last(element[field_name])
end

function (element::Element)(ip, time::Float64=0.0)
    return get_basis(element, ip, time)
end

"""
Examples

julia> el = Element(Quad4, [1, 2, 3, 4]);

julia> el([0.0, 0.0], 0.0, 1)
1x4 Array{Float64,2}:
 0.25  0.25  0.25  0.25

julia> el([0.0, 0.0], 0.0, 2)
2x8 Array{Float64,2}:
 0.25  0.0   0.25  0.0   0.25  0.0   0.25  0.0
 0.0   0.25  0.0   0.25  0.0   0.25  0.0   0.25

"""
function (element::Element)(ip, time::Float64, dim::Int)
    dim == 1 && return get_basis(element, ip, time)
    Ni = vec(get_basis(element, ip, time))
    N = zeros(dim, length(element)*dim)
    for i=1:dim
        N[i,i:dim:end] += Ni
    end
    return N
end

function (element::Element)(ip, time::Float64, ::Type{Val{:Jacobian}})
    X = element("geometry", time)
    dN = get_dbasis(element, ip, time)
    nbasis = length(element)
    if isa(X.data, Vector)
        J = sum([kron(dN[:,i], X[i]') for i=1:nbasis])
    else
        c = get_connectivity(element)
        J = sum([kron(dN[:,i], X[c[i]]') for i=1:nbasis])
    end
    return J
end

function (element::Element)(ip, time::Float64, ::Type{Val{:detJ}})
    J = element(ip, time, Val{:Jacobian})
    n, m = size(J)
    if n == m  # volume element
        return det(J)
    end
    JT = transpose(J)
    if size(JT, 2) == 1  # boundary of 2d problem, || ∂X/∂ξ ||
        return norm(JT)
    else # manifold on 3d problem, || ∂X/∂ξ₁ × ∂X/∂ξ₂ ||
        return norm(cross(JT[:,1], JT[:,2]))
    end
end

function (element::Element)(ip, time::Float64, ::Type{Val{:Grad}})
    J = element(ip, time, Val{:Jacobian})
    return inv(J)*get_dbasis(element, ip, time)
end

function (element::Element)(field_name::String, ip, time::Float64, ::Type{Val{:Grad}})
    return element(ip, time, Val{:Grad})*element[field_name](time)
end

function (element::Element)(field_name::String, ip, time::Float64)
    field = element[field_name]
    return element(field, ip, time)
end

function (element::Element)(field::DCTI, ip, time::Float64)
    return field.data
end

function (element::Element)(field::DCTV, ip, time::Float64)
    return field(time).data
end

function (element::Element)(field::CVTV, ip, time::Float64)
    return field(ip, time)
end

function (element::Element)(field::Field, ip, time::Float64)
    field_ = field(time)
    basis = element(ip, time)
    n = length(element)
    if isa(field_.data, Vector)
        m = length(field_)
        if n != m
            error("Error when trying to interpolate field $field at coords $ip and time $time: element length is $n and field length is $m, f = Nᵢfᵢ makes no sense!")
        end
        return sum([field_[i]*basis[i] for i=1:n])
    else
        c = get_connectivity(element)
        return sum([field_[c[i]]*basis[i] for i=1:n])
    end
end

function size(element::Element, dim)
    return size(element)[dim]
end

""" Update element field based on a dictionary of nodal data and connectivity information.

Examples
--------
julia> data = Dict(1 => [0.0, 0.0], 2 => [1.0, 2.0])
julia> element = Seg2([1, 2])
julia> update!(element, "geometry", data)

As a result element now have time invariant (variable) vector field "geometry" with data ([0.0, 0.0], [1.0, 2.0]).

"""
function update!{E}(element::Element{E}, field_name::AbstractString, data::Dict)
    #element[field_name] = Field(data)
    element_id = element.id
    local_connectivity = get_connectivity(element)
    for i in local_connectivity
        if !haskey(data, i)
            ndata = length(data)
            critical("Unable to set field data $field_name for element $E with
            id $element_id and connectivity $local_connectivity: no data for
            node id $i found. Length of data dict = $ndata")
        end
    end
    local_data = [data[i] for i in local_connectivity]
    element[field_name] = local_data
end

function update!{K,V}(element::Element, field_name, data::Pair{Float64, Dict{K, V}})
    time, field_data = data
    element_data = V[field_data[i] for i in get_connectivity(element)]
    update!(element, field_name, time => element_data)
end

function update!(element::Element, field_name::AbstractString, data::Pair{Float64, Vector{Any}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name::AbstractString, data::Pair{Float64, Vector{Int64}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name::AbstractString, data::Pair{Float64, Vector{Float64}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name::AbstractString, data::Pair{Float64, Vector{Vector{Float64}}})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name::AbstractString, data::Pair{Float64, Float64})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        element[field_name] = data
    end
end

function update!(element::Element, field_name::AbstractString, data::Union{Float64, Vector})
    if haskey(element, field_name)
        update!(element[field_name], data)
    else
        if length(data) != length(element)
            update!(element, field_name, DCTI(data))
        else
            element[field_name] = data
        end
    end
end

function update!(element::Element, datas::Pair...)
    for (field_name, data) in datas
        if haskey(element, field_name)
            update!(element[field_name], data)
        else
            element[field_name] = data
        end
    end
end

function update!(element::Element, field_name::String, data::Function)
    element[field_name] = data
end

function update!(element::Element, field_name::String, field::Field)
    element[field_name] = field
end

function update!(elements::Vector, field_name::String, data)
    for element in elements
        update!(element, field_name, data)
    end
end

""" Check existence of field. """
function haskey(element::Element, field_name::String)
    haskey(element.fields, field_name)
end

function get_connectivity(element::Element)
    return element.connectivity
end

function get_integration_points(element::Element)
    # first time initialize default integration points
    if length(element.integration_points) == 0
        ips = get_integration_points(element.properties)
        element.integration_points = [IP(i, w, xi) for (i, (w, xi)) in enumerate(ips)]
    end
    return element.integration_points
end

""" This is a special case, temporarily change order
of integration scheme mainly for mass matrix.
"""
function get_integration_points(element::Element, change_order::Int)
    order = get_integration_order(element.properties)
    order += change_order
    ips = get_integration_points(element.properties, order)
    return [IP(i, w, xi) for (i, (w, xi)) in enumerate(ips)]
end

""" Return dual basis transformation matrix Ae. """
function get_dualbasis(element::Element, time::Float64, order=1)
    nnodes = length(element)
    De = zeros(nnodes, nnodes)
    Me = zeros(nnodes, nnodes)
    for ip in get_integration_points(element, order)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        De += w*diagm(vec(N))
        Me += w*N'*N
    end
    return De, Me, De*inv(Me)
end

""" Find inverse isoparametric mapping of element. """
function get_local_coordinates(element::Element, X::Vector, time::Float64; max_iterations=10, tolerance=1.0e-6)
    haskey(element, "geometry") || error("element geometry not defined, cannot calculate inverse isoparametric mapping")
    dim = size(element, 1)
    dim == length(X) || error("manifolds not supported.")
    xi = zeros(dim)
    dX = element("geometry", xi, time) - X
    for i=1:max_iterations
        J = element(xi, time, Val{:Jacobian})'
        xi -= J \ dX
        dX = element("geometry", xi, time) - X
        norm(dX) < tolerance && return xi
    end
    info("X = $X, dX = $dX, xi = $xi")
    error("Unable to find inverse isoparametric mapping for element $element for X = $X")
end

""" Test is X inside element. """
function inside{E}(element::Element{E}, X, time)
    xi = get_local_coordinates(element, X, time)
    return inside(E, xi)
end

function calculate_jacobian!{T}(J::Matrix{T}, dN::Matrix{T}, X::Vector{Vector{T}})
    fill!(J, 0.0)
    dim, nnodes = size(dN)
    @assert size(J) == (dim, dim)
    @assert length(X) >= nnodes
    for i=1:nnodes
        Xi = X[i]
        @assert length(Xi) == dim
        for j=1:dim
            for k=1:dim
                @inbounds J[j,k] += dN[j,i]*X[i][k]
            end
        end
    end
    return nothing
end

function grad!{T}(Y::Matrix{T}, dN::Matrix{T}, u::Vector{Vector{T}})
    fill!(Y, 0.0)
    dim, nnodes = size(dN)
    @assert size(Y) == (dim, dim)
    @assert length(u) >= nnodes
    for i=1:nnodes
        ui = u[i]
        @assert length(ui) == dim
        for j=1:dim
            for k=1:dim
                @inbounds Y[j,k] += dN[j,i]*ui[k]
            end
        end
    end
    return nothing
end

""" In place inversion of 3x3 matrix. """
function inv3!{T}(invP::Matrix{T}, P::Matrix{T})::T
    @assert size(P) == (3, 3)
    @assert size(invP) == (3, 3)
    a, b, c, d, e, f, g, h, i = P
    detP = a*(e*i-f*h) + b*(f*g-d*i) + c*(d*h-e*g)
    @assert detP != 0.0
    invP[1] = 1/detP * (e*i - f*h)
    invP[2] = 1/detP * (c*h - b*i)
    invP[3] = 1/detP * (b*f - c*e)
    invP[4] = 1/detP * (f*g - d*i)
    invP[5] = 1/detP * (a*i - c*g)
    invP[6] = 1/detP * (c*d - a*f)
    invP[7] = 1/detP * (d*h - e*g)
    invP[8] = 1/detP * (b*g - a*h)
    invP[9] = 1/detP * (a*e - b*d)
    return detP
end

function element_info!{E,T}(element::Element{E}, ip::IP, time::Float64, N::Matrix{T}, dN::Matrix{T}, J::Matrix{T}, invJ::Matrix{T})
    field = element("geometry")
    if isa(field, DVTI)
        X = field.data
    else
        if time < first(field).time
            X = first(field).data
        elseif time > last(field).time
            X = last(field).data
        else
            for i=reverse(1:length(field))
                if isapprox(field[i].time, time)
                    X = field[i].data
                    break
                end
            end
            # very unlikely but we may need to interpolate
            for i=reverse(2:length(field))
                t0 = field[i-1].time
                t1 = field[i].time
                if t0 < time < t1
                    y0 = field[i-1].data
                    y1 = field[i].data
                    dt = t1-t0
                    X = y0*(1-(time-t0)/dt) + y1*(1-(t1-time)/dt)
                end
            end
        end
    end
    evaluate_basis!(E, ip.coords, N)
    evaluate_dbasis!(E, ip.coords, dN)
    calculate_jacobian!(J, dN, X)
    detJ = inv3!(invJ, J)
    return detJ
end
