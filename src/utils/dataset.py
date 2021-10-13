from collections import defaultdict

def get_annotations_info(annotations_files):
    info = defaultdict(bool)
    parsed = defaultdict(list)
    
    for fn in annotations_files:
        # check if there is train annotations
        if 'train' in fn:
            info['train'] = True
            parsed['train_fn'].append(fn)
        elif 'test' in fn:
            info['test'] = True
            parsed['test_fn'].append(fn)
        elif 'val' in fn:
            info['val'] = True
            parsed['val_fn'].append(fn)
        else:
            info['unknown'] = True
            parsed['unknown_fn'].append(fn)

    return info, parsed