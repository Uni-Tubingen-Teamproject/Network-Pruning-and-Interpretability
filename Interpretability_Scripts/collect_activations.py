"""
Records the most extreme activations for a few chosen layers, for all neurons,
for all validation set images.

This script extracts all known layers of the chosen model, then splits it into
n_chunks chunks, then collects stimuli for one of the chunks.
"""
import argparse
import collections
import logging
import os
import pickle
from typing import Any, Callable, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import torch
from sg_utils import (ModelHookWithAggregation, accuracies,
                      aggregate_activations, get_clip_logits,
                      get_clip_zero_shot_classifier, get_dataloader,
                      get_default_device, get_label_translator,
                      get_layers_from_units_list, get_model_layers,
                      get_relevant_layers, load_model, read_units_file)


@torch.no_grad()
def record_activations_pickle(
    model: torch.nn.Module,
    model_name: str,
    layer_names: Sequence[str],
    target_path: str,
    use_validation_set: bool,
    top_bottom_k: int,
    imagenet_path: str,
    batch_size: int,
    ignore_extreme_image_range: bool,
    store_activations_on_cpu: bool,
    store_activations_as_float16: bool,
    device: Optional[Any] = None,
) -> None:
    """
    Records the activation achieved by every image at every neuron, writes to npy-file.

    :param model: the pytorch model
    :param model_name: the name of the model as str
    :param layer_names: the names of the layers
    :param target_path: path to folder where CSV should be stored
    :param use_validation_set: whether to source exemplars from the validation set
    :param top_bottom_k: how many of the top and bottom activations to record
    :param imagenet_path: path to the imagenet dataset
    :param batch_size: batch size for the dataloader
    :param ignore_extreme_image_range: whether to ignore the top_bottom_k images
        with the most extreme activations
    :param store_activations_on_cpu: whether to store activations on CPU
    :param store_activations_as_float16: whether to store activations as float16
    :param device: the device to use
    """
    logging.info(
        f"Recording activations for layers {
            layer_names}, will write to pkl-file."
    )
    # `results` maps a layer name to a dictionary.
    # This dict maps "activations" to a n_imgs x n_units numpy array
    # and "paths" to a list of strings of length n_imgs

    def get_name(fname) -> str:
        # extracts the filename from a full filepath (to train image)
        return fname[len(imagenet_path):]

    source = "val" if use_validation_set else "train"
    use_webdataset = imagenet_path.endswith(".tar")
    ds_path = imagenet_path if use_webdataset else os.path.join(
        imagenet_path, source)
    dataloader = get_dataloader(
        ds_path,
        model_name,
        batch_size=batch_size,
        return_indices=False,
        use_webdataset=use_webdataset,
    )
    image_paths = []

    # for clip models, we need to construct a zero-shot classifier
    if "clip" in model_name and not model_name.startswith("timm"):
        classifier = get_clip_zero_shot_classifier(
            model, model_name, device=device)

    if model_name == "googlenet":
        label_translator = get_label_translator()

    # store the number of correct and total samples for evaluating performance
    correct_samples = 0
    total_samples = 0

    # These are dictionaries mapping layer names to numpy arrays of shape
    # (2 * max(top_bottom_k, batch_size), n_units)
    # In the end, only the first top_bottom_k rows will be used and the rest will
    # only be used as temporary storage.
    min_activations = {}
    max_activations = {}
    min_indices = {}
    max_indices = {}

    def get_aggregation_fn(layer_name):
        @torch.no_grad()
        def inner(x: torch.Tensor) -> torch.Tensor:
            x = aggregate_activations(x.detach(), layer_name, len(batch))
            if store_activations_on_cpu:
                x = x.cpu()
            if store_activations_as_float16:
                x = x.half()
            return x

        return inner

    # We need to skip some layers as they are not used during inference.
    skipped_layers = []

    hook = ModelHookWithAggregation(
        model, layer_names=layer_names, get_aggregation_fn=get_aggregation_fn
    )
    with hook as hook_fn:
        for batch_number, (batch, labels, paths) in enumerate(dataloader):
            logging.info(f"Batch {batch_number}")
            indices = np.arange(len(paths)) + len(image_paths)
            if not use_webdataset:
                paths = [get_name(p) for p in paths]
            image_paths.extend(paths)
            indices = torch.tensor(
                np.array(indices), dtype=torch.int32, device=device)

            batch = batch.to(device)
            labels = labels.to(device)

            hook.clear_features()

            if "clip" in model_name and not model_name.startswith("timm"):
                logits = get_clip_logits(model, classifier, batch)
            if model_name == "sparse_googlenet_v1":
                logits = model(batch).logits
            else:
                logits = model(batch)

            del batch

            if model_name == "googlenet":
                assert logits.shape[-1] == 1008
                logits = logits[:, :1000]
                labels = label_translator(labels)

            # for getting the accuracy
            preds = torch.argmax(logits, -1)
            correct_samples += (preds == labels).sum()
            total_samples += labels.shape[0]

            current_start_idx = batch_number * batch_size
            for i, layer_name in enumerate(layer_names):
                if layer_name in skipped_layers:
                    continue

                try:
                    activations = hook_fn(layer_name)
                except RuntimeError as e:
                    skipped_layers.append(layer_name)
                    if "No activations were recorded" in str(e):
                        logging.warning(
                            f"Skipping layer {layer_name} because no "
                            "activations were recorded."
                        )
                        continue
                    else:
                        raise e

                if batch_number == 0:
                    max_activations[layer_name] = torch.empty(
                        (top_bottom_k + batch_size, activations.shape[1]),
                        device=activations.device,
                        dtype=activations.dtype,
                    )
                    min_activations[layer_name] = torch.empty(
                        (top_bottom_k + batch_size, activations.shape[1]),
                        device=activations.device,
                        dtype=activations.dtype,
                    )
                    max_indices[layer_name] = -1 * torch.ones(
                        (top_bottom_k + batch_size, activations.shape[1]),
                        dtype=torch.int32,
                        device=activations.device,
                    )
                    min_indices[layer_name] = -1 * torch.ones(
                        (top_bottom_k + batch_size, activations.shape[1]),
                        dtype=torch.int32,
                        device=activations.device,
                    )

                if current_start_idx <= top_bottom_k:
                    max_activations[layer_name][
                        current_start_idx: current_start_idx + batch_size
                    ] = activations
                    min_activations[layer_name][
                        current_start_idx: current_start_idx + batch_size
                    ] = activations
                    max_indices[layer_name][
                        current_start_idx: current_start_idx + batch_size
                    ] = indices[:, None]
                    min_indices[layer_name][
                        current_start_idx: current_start_idx + batch_size
                    ] = indices[:, None]
                else:
                    # Replace second block of activations with new activations
                    max_activations[layer_name][top_bottom_k:] = activations
                    min_activations[layer_name][top_bottom_k:] = activations
                    max_indices[layer_name][top_bottom_k:] = indices[:, None]
                    min_indices[layer_name][top_bottom_k:] = indices[:, None]

                    # Sort activations and paths. After this block, the first
                    # top_bottom_k activations and paths will be the top activations
                    # we are looking for.
                    min_idxs = torch.argsort(
                        min_activations[layer_name], dim=0, descending=False
                    )
                    max_idxs = torch.argsort(
                        max_activations[layer_name], dim=0, descending=True
                    )
                    max_activations[layer_name] = max_activations[layer_name][
                        max_idxs, np.arange(
                            max_activations[layer_name].shape[1])
                    ]
                    min_activations[layer_name] = min_activations[layer_name][
                        min_idxs, np.arange(
                            min_activations[layer_name].shape[1])
                    ]
                    max_indices[layer_name] = max_indices[layer_name][
                        max_idxs, np.arange(
                            min_activations[layer_name].shape[1])
                    ]
                    min_indices[layer_name] = min_indices[layer_name][
                        min_idxs, np.arange(
                            min_activations[layer_name].shape[1])
                    ]

                del activations

    if np.any(max_indices == -1) or np.any(min_indices == -1):
        raise ValueError(
            "Some indices were not filled in. This can happen if "
            "top_bottom_k is set too high for the dataset size (at most "
            "half of the dataset)."
        )

    non_skipped_layer_names = [
        l for l in layer_names if l not in skipped_layers]

    max_activations = {
        k: max_activations[k][:top_bottom_k].cpu().numpy() for k in max_activations
    }
    min_activations = {
        k: min_activations[k][:top_bottom_k].cpu().numpy() for k in min_activations
    }
    max_indices = {k: max_indices[k][:top_bottom_k].cpu().numpy()
                   for k in max_indices}
    min_indices = {k: min_indices[k][:top_bottom_k].cpu().numpy()
                   for k in min_indices}

    n_samples = max_activations[non_skipped_layer_names[0]].shape[0]
    if ignore_extreme_image_range:
        ignore_extreme_image_start_idx, ignore_extreme_image_end_idx = [
            int(x) for x in ignore_extreme_image_range.split(":")
        ]
        if ignore_extreme_image_start_idx < 0:
            ignore_extreme_image_start_idx += n_samples
        if ignore_extreme_image_end_idx < 0:
            ignore_extreme_image_end_idx += n_samples
        non_ignored_indices = np.concatenate(
            (
                np.r_[:ignore_extreme_image_start_idx],
                np.r_[ignore_extreme_image_end_idx:n_samples],
            )
        )
        max_activations = {
            k: max_activations[k][non_ignored_indices] for k in max_activations
        }
        min_activations = {
            k: min_activations[k][non_ignored_indices] for k in min_activations
        }
        max_indices = {k: max_indices[k]
                       [non_ignored_indices] for k in max_indices}
        min_indices = {k: min_indices[k]
                       [non_ignored_indices] for k in min_indices}

    activations = {
        k: np.concatenate([min_activations[k], max_activations[k]], axis=0)
        for k in non_skipped_layer_names
    }

    # Convert dtype to save storage space
    activations = {k: activations[k].astype(np.float16) for k in activations}

    # indices = {k: np.concatenate([min_indices[k], max_indices[k]], axis=0)
    #           for k in non_skipped_layer_names}
    image_paths = np.array(image_paths, dtype=np.string_)

    max_paths = {k: image_paths[max_indices[k]] for k in max_indices}
    min_paths = {k: image_paths[min_indices[k]] for k in min_indices}

    paths = {
        k: np.concatenate([min_paths[k], max_paths[k]], axis=0)
        for k in non_skipped_layer_names
    }

    # Save image paths map to disk
    # with open(os.path.join(target_path, "image_paths.pkl"), "wb") as f:
    #    pickle.dump(image_paths, f)

    # Create activations.pkl-file for every layer
    for layer in non_skipped_layer_names:
        # construct descriptive file name
        filename = f"{layer}.pkl"
        full_path = os.path.join(target_path, filename)

        logging.info(f"Writing results to file for layer {layer}")
        with open(full_path, "wb") as f:
            pickle.dump(
                {"activations": activations[layer], "paths": paths[layer]}, f)

    # making sure that validation set accuracy had the right value
    acc = correct_samples / total_samples
    logging.info(
        f"{model_name} achieved {
            acc * 100}% {'validation' if use_validation_set else 'training'} set accuracy."
    )

    if use_validation_set:
        if model_name in accuracies:
            eps = 0.01
            assert (
                accuracies[model_name] -
                eps <= acc <= accuracies[model_name] + eps
            ), f"Accuracy ({acc}) was not within tolerance of {accuracies[model_name]}"
        else:
            logging.warning(
                "No accuracy for this model was found in the accuracies dict."
            )


def main(
    model: torch.nn.Module,
    model_name: str,
    model_checkpoint: Optional[str],
    layer_names: Sequence[str],
    use_validation_set: bool,
    top_bottom_k: int,
    imagenet_path: str,
    output_path: str,
    batch_size: int,
    ignore_extreme_image_range: bool,
    target_path: Optional[str] = None,
    store_activations_on_cpu: bool = False,
    store_activations_as_float16: bool = False,
    device: Optional[Any] = None,
) -> None:
    """
    Loads model, then feeds all ImageNet validation set images through the model,
    making sure that
    the validation set performance matches the reported performance. Then, writes a
    pkl file for every chosen layer of the network, in which the activations for
    all top-/bottom-k images at this layer are recorded.

    :param model: the model
    :param model_name: the model name
    :param model_checkpoint: the path to the model checkpoint
    :param layer_names: the names of the layers
    :param use_validation_set: whether to collect stimuli from the validation set
    :param top_bottom_k: the number of top/bottom images to record
    :param imagenet_path: the path to the ImageNet dataset
    :param output_path: the path to the output directory
    :param batch_size: the batch size
    :param ignore_extreme_image_range:
    :param target_path: the path to the target directory
    :param store_activations_on_cpu: whether to store activations on CPU
    :param store_activations_as_float16: whether to store activations as float16
    :param device: the device to use
    """

    # create target directory
    simplified_model_name = (
        model_name.replace("/", "_").replace(":", "_").replace("__", "_")
    )

    if model_checkpoint is not None:
        model_checkpoint = model_checkpoint.replace("//", "/")
        simplified_model_name += "_" + "_".join(
            model_checkpoint.split("/")[-2:]
        ).replace(".pth", "").replace(".pth.tar", "")

    if target_path is None:
        target_path = os.path.join(output_path, simplified_model_name)
        logging.info(f"Writing output to target path {target_path}")
    os.makedirs(target_path, exist_ok=True)

    # check if the model actually has this layer, and no activation-file already exists
    all_layers = get_model_layers(model)
    print(all_layers)
    remaining_layer_names = []
    for layer_name in layer_names:
        assert layer_name in all_layers, f"Model {
            model_name} has no layer {layer_name}"

        # if the file with activations already exists, skip this layer
        suffix = ".pkl"
        if not os.path.exists(os.path.join(target_path, layer_name + suffix)):
            remaining_layer_names.append(layer_name)

    logging.info(
        f"Found {len(remaining_layer_names)} remaining layers: {
            remaining_layer_names}"
    )

    if len(remaining_layer_names) == 0:
        logging.info("All activation files already exist, skipping.")
        return

    record_activations_pickle(
        model,
        model_name,
        remaining_layer_names,
        target_path,
        use_validation_set,
        top_bottom_k,
        imagenet_path,
        batch_size,
        ignore_extreme_image_range,
        store_activations_on_cpu,
        store_activations_as_float16,
        device,
    )


def get_subset(layers: Sequence[str], chunks: int, idx: int) -> Sequence[str]:
    """
    Splits a list of layers into chunks and returns one of them.

    :param layers: list of str, layers of the model
    :param chunks: how many chunks to create
    :param idx: int, index of the chunk to use here
    """

    # how many layers are in one chunk?
    n = int(np.ceil(len(layers) / chunks))

    # get chunks
    layer_sets = [layers[i: i + n] for i in range(0, len(layers), n)]

    return layer_sets[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Which model to use. Supported models are {0} or all of timm.".format(
            list(accuracies.keys())
        ),
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--use_validation_set",
        action="store_true",
        help="Whether to use the validation set.",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=1,
        help="How many chunks to create, see module docstring.",
    )
    parser.add_argument(
        "--layer_chunk",
        type=int,
        default=0,
        help="Index of layer chunk to use, see module docstring.",
    )
    parser.add_argument(
        "--n-extreme-images",
        type=int,
        help="For how many of the top/bottom-k images the activations should be recorded.",
    )
    parser.add_argument(
        "--ignore-extreme-image-range",
        type=str,
        default=None,
        help="Format: start index:end index. If provided, the top/bottom images in "
        "this range will not be saved to the disk.",
    )
    # If activations should only be collected for the units we know we will need, pass units_file.
    parser.add_argument(
        "--units_file",
        type=str,
        default=None,
        required=False,
        help="Optionally: Path to units file that should be used to select layers.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        help="Passing layers explicitly"
    )
    parser.add_argument(
        "--imagenet-path", type=str, required=True, help="Path to ImageNet dataset."
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for forward passes."
    )
    parser.add_argument(
        "--store-activations-on-cpu",
        action="store_true",
        help="Whether to store activations on CPU.",
    )
    parser.add_argument(
        "--store-activations-as-float16",
        action="store_true",
        help="Whether to store activations as float16.",
    )

    args = parser.parse_args()

    return args


# MODEL=sparse_googlenet_v1
# python3 collect_extreme_activations.py --model-checkpoint /fast/tklein/transparent_out/legendary-goat-614/legendary-goat-614_final --imagenet-path="/is/cluster/shared/imagenet" --output-path="/fast/tklein/transparent_out/activations/activations-hard-15k" --n-extreme-images=15000 --ignore-extreme-image-range="-14750:-250" --model=sparse_googlenet_v1 --batch-size=1024 --store-activations-on-cpu --store-activations-as-float16
if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:  %(message)s",
        level=logging.INFO,  # Don't log DEBUG, but INFO, WARNING, ERROR and CRITICAL
    )

    args = parse_args()

    device = get_default_device()

    # get the requested model
    target_model = load_model(args.model, args.model_checkpoint, device=device)

    # get units from file, extract all layers of interest
    if args.layers is not None:
        logging.info("Selecting chosen layers.")
        layers = args.layers
    elif args.units_file is None:
        logging.info("Selecting all layers.")
        layers = get_relevant_layers(
            target_model, args.model, strict_mode=False)
    else:
        logging.info(f"Selecting layers as needed by {args.units_file}.")
        units = read_units_file(args.units_file)
        layers = get_layers_from_units_list(units)

    logging.info(f"Found {len(layers)} relevant layers: {layers}")

    # only choose one subset of this list of layers
    layers = get_subset(layers, args.n_chunks, args.layer_chunk)

    logging.info(f"Will collect activations for {
                 len(layers)} layers: {layers}")

    while True:
        try:
            main(
                model=target_model,
                model_name=args.model,
                model_checkpoint=args.model_checkpoint,
                layer_names=layers,
                use_validation_set=args.use_validation_set,
                top_bottom_k=args.n_extreme_images,
                imagenet_path=args.imagenet_path,
                output_path=args.output_path,
                batch_size=args.batch_size,
                ignore_extreme_image_range=args.ignore_extreme_image_range,
                store_activations_on_cpu=args.store_activations_on_cpu,
                store_activations_as_float16=args.store_activations_as_float16,
                device=device
            )

            break
        except torch.cuda.OutOfMemoryError as e:
            logging.error(
                "CUDA out of memory error, retrying with smaller batch size "
                "({0} instead of {1}).".format(
                    args.batch_size // 2, args.batch_size)
            )
            args.batch_size = args.batch_size // 2

            if args.batch_size == 0:
                logging.error("Batch size is 0, exiting.")
