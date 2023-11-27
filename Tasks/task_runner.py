from Tasks import task_0
from Tasks import task_1
from Tasks import task_2
from Tasks import task_3
from Tasks import task_4

task_runner_map = {
    0: task_0.Execute,
    1: task_1.Execute,
    2: task_2.Execute,
    3: task_3.Execute,
    4: task_4.Execute,
}

def Run(task_id, arguments, db):
    task_runner_map[task_id](arguments, db)