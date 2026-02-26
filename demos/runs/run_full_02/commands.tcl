
proc __reply {msg} {
    set f [open {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/reply.txt} w]
    puts $f $msg
    close $f
}

screenshot {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/step_frame.png}
__reply "RID=69f1df67be ok:screenshot file=C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/step_frame.png t=[clock milliseconds]"
