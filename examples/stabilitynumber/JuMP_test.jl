
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((2, true),),
    ((2, false),),
    ]
insts["fast"] = [
    ((20, true),),
    ((20, false),),
    ((30, true),),
    ((30, false),),
    ]
insts["slow"] = [
    ((40, true), nothing, relaxed_tols),
    ((40, false), nothing, relaxed_tols),
    ]
insts["various"] = vcat(insts["fast"], insts["slow"])
return (StabilityNumber, insts)
