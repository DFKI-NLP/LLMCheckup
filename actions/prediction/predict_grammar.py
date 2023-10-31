COVID_GRAMMAR = r"""
?start: action
action: operation done

done: " [e]"

operation: refute | support

refute: " REFUTED"

support: " SUPPORTED"
"""