COVID_GRAMMAR = r"""
?start: action
action: operation done

done: " [e]"

operation: refute | support

refute: " REFUTED"

support: " SUPPORTED"
"""

ECQA_GRAMMAR = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 | op2 | op3 | op4 | op5

op1: " 1"
op2: " 2"
op3: " 3"
op4: " 4"
op5: " 5"
"""
