try:
    from common_python.pred_serv_orm import LongShortGroup

    print("Local package import successful!")
except ImportError as e:
    print(f"Failed to import local package: {e}")
    exit(1)
