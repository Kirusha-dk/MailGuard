"""Engineering smoke on four invented examples, never a quality benchmark."""
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from v82_neural import (CONFIG, NeuralMail, begin_session, new_progress, optimizer_and_scheduler,
                        restore_checkpoint, run_steps, save_checkpoint, trainable_state)


def main():
    torch.set_num_threads(3); torch.use_deterministic_algorithms(True)
    root = Path('.cache/v82/backbone'); output = Path('reports/v82-pretrained-smoke.json')
    config = dict(CONFIG, batch_size=2, max_epochs=4)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    texts = [('Please review the agenda before our meeting tomorrow.', 0),
             ('Your software subscription receipt is attached. Thank you for your payment.', 0),
             ('Congratulations! Claim your cash prize now by sending a processing fee.', 1),
             ('Your account will close today. Verify your password at this link immediately.', 1)]
    rows = [dict(identity=str(i), text=text, y=y) for i, (text, y) in enumerate(texts)]

    def make():
        torch.manual_seed(82)
        model = NeuralMail(AutoModel.from_pretrained(root, use_safetensors=True,
                          local_files_only=True, attn_implementation='eager'), config)
        optimizer, scheduler = optimizer_and_scheduler(model, 20, config)
        return model, optimizer, scheduler

    def advance(bundle, progress):
        run_steps(bundle[0], tokenizer, rows, [1.] * 4, bundle[1], bundle[2], progress, config)

    def frozen_sha(model):
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                digest.update(name.encode()); digest.update(parameter.detach().numpy().tobytes())
        return digest.hexdigest()

    full = make(); original_sha = frozen_sha(full[0]); before = trainable_state(full[0])
    progress = new_progress(); begin_session(progress, 'engineering-only', 6); advance(full, progress)
    after = trainable_state(full[0]); frozen_unchanged = original_sha == frozen_sha(full[0])
    split = make(); progress = new_progress(); begin_session(progress, 'engineering-only', 3); advance(split, progress)
    checkpoint = Path('.cache/v82/smoke-checkpoint.pt'); metadata = dict(engineering_only=True, config=config)
    save_checkpoint(checkpoint, *split, progress, metadata)
    resumed = make(); progress = restore_checkpoint(checkpoint, *resumed, metadata)
    progress['active_target'] = 6; advance(resumed, progress)
    difference = max(float((after[k] - trainable_state(resumed[0])[k]).abs().max()) for k in after)
    adapters = [k for k in after if k.endswith(('.a', '.b'))]
    report = dict(pretrained_model=CONFIG['model'], revision=CONFIG['revision'], engineering_test_only=True,
                  synthetic_examples=4, quality_gain_demonstrated=False, optimizer_steps=6,
                  trainable_parameters=sum(p.numel() for p in full[0].parameters() if p.requires_grad),
                  changed_internal_adapter_tensors=sum(not torch.equal(after[k], before[k]) for k in adapters),
                  internal_adapter_delta_l2=math.sqrt(sum(float((after[k] - before[k]).square().sum()) for k in adapters)),
                  base_weight_unchanged=frozen_unchanged, resume_max_absolute_difference=difference)
    if not frozen_unchanged or difference != 0 or report['changed_internal_adapter_tensors'] == 0:
        raise AssertionError(report)
    output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
