import importlib
import sys

_method_mapping = {
    'FigStep': 'multi_attacks.figstep',
    'CS-DJ': ('multi_attacks.cs_dj', 'CSDJAttack'),
    'Hades': 'multi_attacks.hades',
    'JOOD': ('multi_attacks.JOOD', 'JOODAttack'),
    'JPS': ('multi_attacks.jps', 'JPS'),
    'MML': ('multi_attacks.mml', 'MML'),
    'BestOfN': 'multi_attacks.best_of_n',
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
    entry = _method_mapping[method]
    module_path, class_name = entry if isinstance(entry, tuple) else (entry, method)
    print("module_path")
    print(module_path)
    
    # 기존 모듈이 로드되어 있다면 제거
    if module_path in sys.modules:
        del sys.modules[module_path]
    
    module = importlib.import_module(module_path)
    method_class = getattr(module, class_name)
    print("method_class")
    print(method_class)
    return method_class    

def init_method(method_class,method_config):
    from data_paths import resolve_path
    method_config = dict(method_config)
    for key in ('image_path', 'image_root', 'target_path', 'harmful_image_dir',
                'harmless_image_dir', 'prompt_dir', 'font_path'):
        value = method_config.get(key)
        if isinstance(value, str) and value.startswith(('data/', './data/')):
            method_config[key] = resolve_path(value)
    output = method_class(**method_config)
    #output.get_model_and_tokenizer(model_name)
    print("output")
    print(output)
    return output
