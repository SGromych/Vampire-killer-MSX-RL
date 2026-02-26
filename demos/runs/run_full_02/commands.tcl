
proc __reply {msg} {
    set f [open {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/reply.txt} w]
    puts $f $msg
    close $f
}

screenshot {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/step_frame.png}
__reply "RID=e294b3c5ca ok:screenshot file=C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/step_frame.png t=[clock milliseconds]"
