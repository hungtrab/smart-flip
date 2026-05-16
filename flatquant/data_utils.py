import os
import pickle
import datasets
import random
import transformers


C4_TRAIN_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
C4_VALIDATION_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split='train')
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    split = "validation" if eval_mode else "train"
    url = C4_VALIDATION_URL if eval_mode else C4_TRAIN_URL
    dataset = datasets.load_dataset(
        "json",
        data_files={split: url},
        split=split,
        streaming=True,
    )

    if eval_mode:
        texts = []
        for item in dataset:
            text = item.get("text", "").strip()
            if not text:
                continue
            texts.append(text)
            if len(texts) >= 1100:
                break
        valenc = tokenizer(' '.join(texts), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc

    trainloader = []
    for item in dataset:
        text = item.get("text", "").strip()
        if not text:
            continue
        trainenc = tokenizer(text, return_tensors='pt')
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        max_start = trainenc.input_ids.shape[1] - seqlen
        start = random.randint(0, max_start)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        if len(trainloader) >= nsamples:
            break

    if len(trainloader) < nsamples:
        raise RuntimeError(f"Unable to collect {nsamples} C4 samples; only found {len(trainloader)} usable documents")
    return trainloader


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seqlen, tokenizer):
    traindata = datasets.load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    args, name, tokenizer, nsamples=128, seqlen=2048, eval_mode=False
):
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seqlen, tokenizer)

    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset
