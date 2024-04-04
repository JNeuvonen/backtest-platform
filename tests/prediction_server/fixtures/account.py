def create_acc_body(name: str, max_debt_ratio: float):
    return {"name": name, "max_debt_ratio": max_debt_ratio}


def create_master_acc():
    return create_acc_body(name="master_acc", max_debt_ratio=1.0)
