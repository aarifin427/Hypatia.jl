
insts = Dict()
insts["minimal"] = [
    ((1, 1, 1, 2, true, true, false),),
    ((1, 1, 1, 2, true, false, true),),
    ((1, 1, 1, 2, false, true, false),),
    ((1, 1, 1, 2, false, false, true),),
    ((1, 1, 1, 2, false, false, false),),
    ]
insts["fast"] = [
    ((2, 1, 2, 2, true, true, false),),
    ((2, 1, 2, 2, false, false, true),),
    ((2, 1, 2, 2, false, false, false),),
    ((2, 1, 1, 3, true, false, true),),
    ((2, 1, 1, 3, false, true, false),),
    ((4, 2, 3, 10, true, true, false),),
    ((4, 2, 3, 10, true, false, true),),
    ]
insts["slow"] = [
    ((4, 2, 3, 10, false, true, false),),
    ((4, 2, 3, 10, false, false, false),),
    ]
return (PolyNormJuMP, insts)
