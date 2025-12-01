# --------------------------- Database Errors ---------------------------
class DBError(Exception):
    pass


class ConstantDoesNotExist(DBError):
    message_prefix = 'Constant does not exist: '
    pass


class FunctionAlreadyExists(DBError):
    default_msg = 'Function already exists: '
    pass


class FunctionDoesNotExist(DBError):
    default_msg = 'Function does not exist: '
    pass


class ConstantAlreadyExists(DBError):
    default_msg = 'Constant already exists: '
    pass


class NoSuchInspirationFunction(DBError):
    default_msg = 'No such inspiration function: '
    pass


# --------------------------- JSON Errors ---------------------------
class JSONError(Exception):
    pass


class FormattingError(JSONError):
    bad_format_msg = """Invalid JSON file, please check the syntax of the file:
    {
        "command": <"update" or "replace" or "insert"">,
        "data":
        [   
            {
                "constant": <constant>,
                "data": {
                    "<type config>": <type class>,
                    "<data config>": <data>,
                    <<<optional>>> "kwargs": {<kwargs>}
                },
            },
            {...}
        ]
    }
    """
    pass


class MissingPath(JSONError):
    default_msg = 'Missing path to JSON file'
