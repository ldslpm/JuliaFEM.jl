# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

typealias Node Vector{Float64}

abstract AbstractPoint

immutable Point{P<:AbstractPoint}
    id :: Int
    weight :: Float64
    coords :: Tuple{Vararg{Float64}}
    fields :: Dict{String, Field}
    properties :: P
end

function setindex!{T}(point::Point, val::Pair{Float64, T}, field_name)
    point.fields[field_name] = Field(val)
end

function getindex(point::Point, field_name)
    return point.fields[field_name]
end

function getindex(point::Point, idx::Int)
    return point.coords[idx]
end

function haskey(point::Point, field_name)
    return haskey(point.fields, field_name)
end

function (point::Point)(field_name, time=0.0)
    point.fields[field_name](time).data
end

function update!{T}(point::Point, field_name, val::Pair{Float64, T})
    if haskey(point, field_name)
        update!(point[field_name], val)
    else
        point[field_name] = val
    end
end

#= TODO: in future
type Node <: AbstractPoint
end

type MaterialPoint <: AbstractPoint
end
=#

type IntegrationPoint <: AbstractPoint
end

typealias IP Point{IntegrationPoint}

function IP(id::Int, weight::Float64, coords::Tuple{Vararg{Float64}})
    return IP(id, weight, coords, Dict(), IntegrationPoint())
end
