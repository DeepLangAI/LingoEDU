import os

cur_dir = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(cur_dir, "solver_user.txt"), "r", encoding="utf8") as f:
    SolverPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "solver_prefix.txt"), "r", encoding="utf8") as f:
    SolverPrompt_Assistant_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "select_user.txt"), "r", encoding="utf8") as f:
    SelectPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "select_prefix.txt"), "r", encoding="utf8") as f:
    SelectPrompt_Assistant_Template = "".join(f.readlines())

