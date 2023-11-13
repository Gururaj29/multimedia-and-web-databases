import argparse
import toml
import textwrap
from Tasks import task_runner
import readline
import storage


### internal methods

def extract_args(arg_parser):
    task_id = None
    args = {}
    for key, value in arg_parser._get_kwargs():
        if key == "task_id":
            task_id = value
        else:
            args[key] = value
    return task_id, args

def run_task(args, db):
    task_id, arguments = extract_args(args)
    task_runner.Run(task_id, arguments, db)

def load_data():
    # load the database for interactive mode
    db = storage.Database()
    return db

NUM_TASKS = 6

with open("cli.toml", "r") as config_file:
    cli_config = toml.load(config_file)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--interactive", action = argparse.BooleanOptionalAction, dest="is_interactive")
subparsers = parser.add_subparsers()

for i in range(NUM_TASKS):
    task_config = cli_config["tasks"][str(i)]
    task_arg_parser = subparsers.add_parser("task%d"%i, 
        help=task_config["usage"], 
        description=textwrap.dedent(task_config["description"]), 
        usage=task_config["usage"], 
    formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in task_config["args"]:
        options = cli_config["enums"][arg["enum"]] if "enum" in arg else None
        flag = "--" + arg["id"]
        if "enum" in arg:
            options = cli_config["enums"][arg["enum"]]
            task_arg_parser.add_argument(flag, type=eval(arg["type"]), help=arg["description"] + ". Allowed options are: " + ', '.join(options), choices=options, metavar='')
        elif "bool" in arg:
            task_arg_parser.add_argument(flag, help = arg["description"],action=argparse.BooleanOptionalAction)
        else:
            task_arg_parser.add_argument(flag, type=eval(arg["type"]), help=arg["description"], metavar='')
    task_arg_parser.set_defaults(task_id = i)

args = parser.parse_args()
db = load_data()

if args.is_interactive:
    # interactive mode
    while True:
        # Command in green
        inp = input("\033[1;32mCommand >>> \033[0m").split()
        
        should_exit = "exit()" in inp or "exit" in inp
        if should_exit:
            break
        
        no_op_cmd = False
        try:
            args = parser.parse_args(inp)
        except SystemExit:
            # argparse exits on 'help' commands, avoiding that by catching SystemExit
            no_op_cmd = True
        
        if not no_op_cmd:
            run_task(args, db)
else:
    run_task(args, db)
