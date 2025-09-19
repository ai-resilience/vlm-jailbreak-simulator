import importlib
import sys

_method_mapping = {
    'FigStep': 'multi_attacks.figstep',
    'Hades': 'multi_attacks.hades',
    'Query_Attack': 'image_attacks.query_attack',
    'Visual_Adv': 'image_attacks.visual_adv',
    'ImgJP': 'image_attacks.imgjp',
    'DeltaJP': 'image_attacks.deltajp'
}

def get_method_class(method):
    """
    Returns the method class given the method name. This is used to access static methods.
    """
    if method not in _method_mapping:
        raise ValueError(f"Can not find method {method}")
    module_path = _method_mapping[method]
    print("module_path")
    print(module_path)
    
    # 기존 모듈이 로드되어 있다면 제거
    if module_path in sys.modules:
        del sys.modules[module_path]
    
    module = importlib.import_module(module_path)
    method_class = getattr(module, method)  # Class name assumed to be the same as method name
    print("method_class")
    print(method_class)
    return method_class    

def init_method(method_class,method_config):

    output = method_class(**method_config)
    #output.get_model_and_tokenizer(model_name)
    print("output")
    print(output)
    return output
