class Accounts:
    MASTER_ACC = "master_acc"


def create_acc_body(name: str, max_debt_ratio: float):
    return {"name": name, "max_debt_ratio": max_debt_ratio}


def create_master_acc():
    return create_acc_body(name=Accounts.MASTER_ACC, max_debt_ratio=1.0)
