# bootstrap.tcl (diagnostic + polling)

set CMD_FILE "C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/commands.tcl"

proc __write_reply {msg} {
    set f [open {C:/Users/sergey.gromov/Documents/AI/RL_msx/demos/runs/run_full_02/reply.txt} w]
    puts $f $msg
    close $f
}

set __poll_count 0
set __last_status_ms 0

proc __poll_commands {} {
    global CMD_FILE
    global __poll_count
    global __last_status_ms

    incr __poll_count

    # Heartbeat every ~500ms — в bootstrap_status, НЕ в reply.txt (не затирать ответы)
    set now [clock milliseconds]
    if {$__last_status_ms == 0 || ($now - $__last_status_ms) > 500} {
        set exists [file exists $CMD_FILE]
        set size 0
        if {$exists} { set size [file size $CMD_FILE] }
        if {[catch {set stf [open "bootstrap_status.txt" w]; puts $stf "ok:bootstrap poll=$__poll_count cmd_exists=$exists cmd_size=$size"; close $stf}] == 0} {}
        set __last_status_ms $now
    }

    if {[file exists $CMD_FILE] && [file size $CMD_FILE] > 0} {
        set err ""
        set code [catch {source $CMD_FILE} err]
        if {$code != 0} {
            __write_reply "err:source poll=$__poll_count msg=$err"
        }
        # truncate after exec
        set f [open $CMD_FILE w]
        close $f
    }

    after 20 __poll_commands
}

__write_reply "ok:bootstrap starting"
after 20 __poll_commands

# Load ROM
carta {C:/Users/sergey.gromov/Documents/AI/RL_msx/VAMPIRE.ROM}

# Настройка джойстика — в catch, чтобы ошибка не ломала загрузку
if {[catch {
    plug joyporta msxjoystick1
    dict set msxjoystick1_config LEFT {keyb LEFT}
    dict set msxjoystick1_config RIGHT {keyb RIGHT}
    dict set msxjoystick1_config UP {keyb UP}
    dict set msxjoystick1_config DOWN {keyb DOWN}
    dict set msxjoystick1_config A {keyb SPACE}
    dict set msxjoystick1_config B {keyb LCTRL}
} err] != 0} {
    # игнорируем — картридж уже загружен
}
