import os
import re
import warnings
from setuptools import setup, find_packages


def get_requirements(fns, envsub: bool = False):
    reg = r'\$(\w+)'
    reqs = []
    for fn in fns:
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Given file {fn} does not exists.')
        with open(fn, 'r') as f:
            for line in f.readlines():
                s = line.strip()
                if s.startswith('#'):
                    continue
                if envsub:
                    for k in re.findall(reg, line):
                        v = os.environ.get(k)
                        if v is None:
                            warnings.warn(
                                f'Environment variable "{k}" is required by "{s}"'
                                f'but not given. Skip'
                            )
                            break
                        s = s.replace('$'+k, v)
                    else:
                        reqs.append(s)
                else:
                    reqs.append(s)
    return reqs


setup(
    name='edge_ocr',
    version='0.1.0',
    description='OCR inference tool on edge',
    install_requires=get_requirements(['requirements/basic.txt']),
    extras_require={
        'onnx': get_requirements(['requirements/onnx.txt', 'requirements/basic.txt']),
        'tensorrt': get_requirements(['requirements/tensorrt.txt', 'requirements/basic.txt']),
        'polygraphy': get_requirements(['requirements/polygraphy.txt', 'requirements/tensorrt.txt', 'requirements/basic.txt']),
        'openvino': get_requirements(['requirements/openvino.txt', 'requirements/basic.txt']),
    },
    packages=find_packages(exclude=['main']),
)
