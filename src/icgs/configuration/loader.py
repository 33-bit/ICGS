"""Explicit JSON input: defaults/profile < file fields < explicit CLI overrides."""
from dataclasses import asdict
import json
from pathlib import Path
from .defaults import instant_policy_original, from_resolved


def merge_known(base, update, prefix=''):
    if not isinstance(update, dict):
        raise ValueError(f'{prefix or "config"} must be an object')
    for key,value in update.items():
        if key not in base: raise ValueError(f'unknown configuration key: {prefix}{key}')
        if isinstance(base[key],dict): merge_known(base[key],value,prefix+key+'.')
        else: base[key]=value


def load_config(path=None, *, overrides=None, entry=None):
    data={} if path is None else json.loads(Path(path).read_text())
    if not isinstance(data,dict) or set(data)-{'schema_version','profile','config'}:
        raise ValueError('unknown configuration document fields')
    if data.get('schema_version',1)!=1: raise ValueError('unsupported configuration schema')
    profile=data.get('profile',entry or 'train')
    if profile not in ('train','fine_tune','eval','deploy','published'):
        raise ValueError(f'unknown configuration profile: {profile}')
    if profile=='published':
        from icgs.artifacts.published import published_config
        base=asdict(published_config())
    else:
        base=asdict(instant_policy_original())
        if profile in ('eval','deploy'):
            base['runtime']['batch_size']=1
            base['runtime']['compile_models']=False
            base['sampling']['steps']=4
        if profile=='fine_tune': base['runtime']['compile_models']=False
    merge_known(base,data.get('config',{}))
    for dotted,value in (overrides or {}).items():
        if value is None: continue
        bits=dotted.split('.')
        update=value
        for bit in reversed(bits): update={bit:update}
        merge_known(base,update)
    for section in ('scene','graph','backbone','action','diffusion','sampling'):
        if base[section]['kind']!='original':
            raise ValueError(f'unknown CLI component ID: {section}.{base[section]["kind"]}; use explicit Python factories')
    if base['diffusion']['scheduler_kind']!='original': raise ValueError('unknown scheduler ID')
    # Explicit file-relative paths; no implicit root configs lookup.
    directory=Path(path).resolve().parent if path else Path.cwd()
    for section,key in (('scene','checkpoint'),('training','save_dir')):
        value=base[section][key]
        if value and not Path(value).is_absolute(): base[section][key]=str(directory/value)
    return from_resolved({'schema_version':1,'config':base})
