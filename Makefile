help:
	@python3 cli.py --help

task_help:
	@python3 cli.py task$t --help

run_task:
	@python3 cli.py task$t $(arg)

run_task_one:
	@python3 cli.py task1 --label=$l --feature_model=$(fm) --k=$k

run_interactive:
	@python3 cli.py --i
