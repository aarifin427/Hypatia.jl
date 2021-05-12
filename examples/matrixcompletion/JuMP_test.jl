
insts = Dict()
insts["minimal"] = [
    ((2, 2),),
    ((2, 2), :ExpPSD),
    ((2, 2), :SOCExpPSD),
    ]
insts["fast"] = [
    ((3, 4),),
    ((3, 4), :ExpPSD),
    ((3, 4), :SOCExpPSD),
    ((10, 3),),
    ((5, 3), :ExpPSD),
    ((5, 3), :SOCExpPSD),
    ]
insts["slow"] = [
    ((2, 12), :SOCExpPSD),
    ((10, 14),),
    ((10, 14), :SOCExpPSD),
    ((2, 40),),
    ((10, 18),),
    ]
insts["various"] = [
    ((10, 20),),
    ((10, 20), :ExpPSD),
    ((10, 20), :SOCExpPSD),
    ((10, 25),),
    ((10, 25), :ExpPSD),
    ((10, 25), :SOCExpPSD),
    ((10, 30),),
    ]
return (MatrixCompletionJuMP, insts)
