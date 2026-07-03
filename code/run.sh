#!/bin/bash
julia -t auto -i -e '
include("plate_hop.jl")
plate.main()'
