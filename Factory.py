import MultiObjectiveProblem as MOOP

def set_problem(prob_name):
    if prob_name == "sch" or prob_name == "SCH":
        return MOOP.SCH