import code
import inspect


def get_debug_answer():
    while True:
        answer = input(
            """
        Do you want to:
        (1) run an iteration of the control loop?
        (2) run this control_loop until completion?
        (3) get dropped in an interactive shell?
        Answer with [1/2/3]
        """
        )
        try:
            answer = int(answer)
        except ValueError:
            pass
        if answer in [1, 2, 3]:
            return answer
        else:
            print(
                "Whatever you typed in is neither 1, 2 or 3. Give it to me straight cheif!"
            )
            continue


# NOTE: this one works, and you can update
# internal varibles, but it's a shitty baren python shell.
# so not the solution i want
# def start_interactive_shell():
#    # Get stack frame for caller
#    stack = inspect.stack()[1]
#    frame = stack.frame
#
#    # Copy locals and globals of caller's stack frame
#    locals_copy = dict(frame.f_locals)
#    globals_copy = dict(frame.f_globals)
#    shell_locals = dict()
#    shell_locals.update(globals_copy)
#    shell_locals.update(locals_copy)
#
#    # Start interactive shell
#    code.interact(local=shell_locals)
#
#    # Delete frame to avoid cyclic references
#    del stack
