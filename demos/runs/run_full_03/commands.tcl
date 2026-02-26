
proc __reply {msg} {
    set f [open {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_03/reply.txt} w]
    puts $f $msg
    close $f
}

screenshot {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_03/step_frame.png}
__reply "RID=69d8c8cc00 ok:screenshot file=C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_03/step_frame.png t=[clock milliseconds]"
