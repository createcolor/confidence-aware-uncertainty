def validation_check(*args):
    for arg in args:
        assert arg.exists(), f"{arg} does not exist"

def mkdirs(*dir_paths):
    for dir_path in dir_paths:
        if not (dir_path).exists():
            (dir_path).mkdir()