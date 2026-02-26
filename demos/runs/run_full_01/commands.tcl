
proc __reply {msg} {
    set f [open {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_01/reply.txt} w]
    puts $f $msg
    close $f
}

screenshot {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_01/step_frame.png}
__reply "RID=38ed61e3d9 ok:screenshot file=C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_01/step_frame.png t=[clock milliseconds]"
