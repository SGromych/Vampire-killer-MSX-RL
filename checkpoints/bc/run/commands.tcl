
proc __reply {msg} {
    set f [open {C:/Users/sergey.gromov/Documents/AI/RL_msx/checkpoints/bc/run/reply.txt} w]
    puts $f $msg
    close $f
}

screenshot {C:/Users/sergey.gromov/Documents/AI/RL_msx/checkpoints/bc/run/step_frame.png}
__reply "RID=750769a4a8 ok:screenshot file=C:/Users/sergey.gromov/Documents/AI/RL_msx/checkpoints/bc/run/step_frame.png t=[clock milliseconds]"
