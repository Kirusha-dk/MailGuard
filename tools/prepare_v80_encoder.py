"""Requantize the pinned English encoder to avoid U8S8 CPU saturation."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from v80_encoder import MODEL, REVISION, VIEW, file_sha

SOURCE_SHA = '4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474'


def prepare(source, target, tokenizer, license_file):
    import onnx
    from onnx import numpy_helper
    if file_sha(source) != SOURCE_SHA:
        raise ValueError('Unexpected original quantized encoder')
    target.mkdir(parents=True, exist_ok=False)
    model = onnx.load(source)
    initializers = {t.name: t for t in model.graph.initializer}
    names = {node.input[1] for node in model.graph.node
             if node.op_type in ('MatMulInteger', 'QAttention') and len(node.input) > 1}
    converted = 0
    for name in sorted(names):
        tensor = initializers.get(name)
        if tensor is None:
            continue
        values = numpy_helper.to_array(tensor)
        if values.dtype != np.int8:
            continue
        if not name.endswith('_quantized'):
            raise ValueError('Unknown signed weight format')
        base = name.removesuffix('_quantized')
        zero = numpy_helper.to_array(initializers[base + '_zero_point'])
        if np.any(zero != 0):
            raise ValueError('Expected symmetric signed weights')
        scale = numpy_helper.to_array(initializers[base + '_scale'])
        rounded = np.clip(np.rint(values.astype(np.float32) / 2), -64, 63).astype(np.int8)
        tensor.CopyFrom(numpy_helper.from_array(rounded, name))
        initializers[base + '_scale'].CopyFrom(numpy_helper.from_array(
            (scale * 2).astype(scale.dtype), base + '_scale'))
        converted += 1
    if converted < 30:
        raise ValueError('Too few signed matrices converted')
    onnx.checker.check_model(model)
    onnx.save(model, target / 'model.onnx')
    shutil.copyfile(tokenizer, target / 'tokenizer.json')
    shutil.copyfile(license_file, target / 'LICENSE')
    manifest = dict(model=MODEL, revision=REVISION, view=VIEW,
                    source_model_sha256=SOURCE_SHA, converted_matrices=converted,
                    modification='signed weights rounded to [-64,63]; scales doubled; independent activation quantization',
                    files={name: file_sha(target / name) for name in ('model.onnx', 'tokenizer.json', 'LICENSE')})
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--license-file', type=Path, required=True)
    a = p.parse_args(); prepare(a.source, a.out, a.tokenizer, a.license_file)
