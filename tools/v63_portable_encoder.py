"""Requantize signed dynamic MatMul weights to a safe 7-bit CPU range."""
import json
import shutil
from pathlib import Path
import numpy as np
from train_v55_multilingual import file_sha
from v61_semantic import Encoder,ONNX


def prepare_encoder(source,target):
    import onnx
    from onnx import numpy_helper
    source,target=Path(source),Path(target)
    target.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if file_sha(source/name)!=digest:raise ValueError('Original encoder changed')
        shutil.copyfile(source/name,target/name)
    path=target/Path(ONNX).name;model=onnx.load(path)
    initializers={t.name:t for t in model.graph.initializer}
    weights={n.input[1] for n in model.graph.node if n.op_type in ('MatMulInteger','QAttention') and len(n.input)>1}
    converted=[]
    for name in sorted(weights):
        if name not in initializers:continue
        tensor=initializers[name];q=numpy_helper.to_array(tensor)
        if q.dtype!=np.int8:continue
        if not name.endswith('_quantized'):raise ValueError('Unrecognized quantized weight naming')
        base=name.removesuffix('_quantized');scale_name=base+'_scale';zero_name=base+'_zero_point'
        if scale_name not in initializers or zero_name not in initializers:raise ValueError('Missing weight quantization parameters')
        z=numpy_helper.to_array(initializers[zero_name])
        if np.any(z!=0):raise ValueError('Expected symmetric signed weight zero point')
        scale=numpy_helper.to_array(initializers[scale_name])
        q7=np.clip(np.rint(q.astype(np.float32)/2),-64,63).astype(np.int8)
        tensor.CopyFrom(numpy_helper.from_array(q7,name))
        initializers[scale_name].CopyFrom(numpy_helper.from_array((scale*2).astype(scale.dtype),scale_name))
        converted.append(dict(name=name,minimum=int(q7.min()),maximum=int(q7.max())))
    if not converted:raise ValueError('No signed MatMul weights found; CPU saturation hypothesis does not apply')
    onnx.checker.check_model(model);onnx.save(model,path)
    manifest['files'][path.name]=file_sha(path)
    manifest['portableQuantization']=dict(version='signed-7bit-v1',sourceModelSha256=file_sha(source/path.name),
                                        convertedWeights=converted,reason='Avoid U8S8 16-bit intermediate saturation on non-VNNI x64 CPUs')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PORTABLE SIGNED WEIGHTS',len(converted),manifest['files'][path.name],flush=True)
    return Encoder(target)
