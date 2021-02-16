from typing import Dict, List, Union

import libcst as cst


def dict(dictionary: Dict) -> cst.Dict:
    return cst.Dict([cst.DictElement(key, value) for key, value in dictionary.items()])


def call(
    func: Union[cst.Name, cst.Attribute], arg_names: Union[cst.Name, List[cst.Name]]
) -> cst.Call:
    if type(arg_names) == cst.Name:
        arg_names = [arg_names]

    return cst.Call(func, [cst.Arg(arg_name) for arg_name in arg_names])


def param(param_name: str):
    return cst.Param(cst.Name(param_name))
