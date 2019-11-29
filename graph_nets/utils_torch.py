# Copyright 2019 Wei-Di Chang and Manfred Diaz. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================
#
# =========================================================
import collections

import torch

from graph_nets import utils_np
from graph_nets import graphs

ALL_FIELDS = graphs.ALL_FIELDS
SENDERS = graphs.SENDERS
RECEIVERS = graphs.RECEIVERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE
NODES = graphs.NODES
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS


def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.

  The N_NODE field is filled if the graph contains a non-`None` NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-`None` RECEIVERS field;
  otherwise, it is set to 0.

  Args:
    data_dict: An input `dict`.

  Returns:
    The data `dict` with number fields.
  """
  dct = data_dict.copy()
  for number_field, data_field in [[N_NODE, NODES], [N_EDGE, RECEIVERS]]:
    if dct.get(number_field) is None:
      if dct[data_field] is not None:
        dct[number_field] = torch.tensor(dct[data_field].size()[0], dtype=torch.int64)
      else:
        dct[number_field] = torch.tensor(0, dtype=torch.int64)
  return dct


def _to_compatible_data_dicts(data_dicts):
    """Convert the content of `data_dicts` to tensors of the right type.

      All fields are converted to `Tensor`s. The index fields (`SENDERS` and
      `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `tf.int32`.

      Args:
        data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and
          values either `None`s, or quantities that can be converted to `Tensor`s.

      Returns:
        A list of dictionaries containing `Tensor`s or `None`s.
      """
    results = []
    for data_dict in data_dicts:
        result = {}
        for k, v in data_dict.items():
            if v is None:
                result[k] = None
            else:
                dtype = torch.long if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else torch.float32
                result[k] = torch.tensor(v, dtype=dtype)
        results.append(result)
    return results


def _compute_stacked_offsets(sizes, repeats):
    """Computes offsets to add to indices of stacked tensors (Tensorflow).

      When a set of tensors are stacked, the indices of those from the second on
      must be offset in order to be able to index into the stacked tensor. This
      computes those offsets.

      Args:
        sizes: A 1D `Tensor` of the sizes per graph.
        repeats: A 1D `Tensor` of the number of repeats per graph.

      Returns:
        A 1D `Tensor` containing the index offset per graph.
      """
    sizes = torch.tensor(sizes[:-1]).type(torch.long)
    offset_values = torch.cumsum(torch.cat([torch.zeros(1).long(), sizes], 0), dim=0)
    return torch.repeat_interleave(offset_values, repeats.long())


def _concatenate_data_dicts(data_dicts):
    """Concatenate a list of data dicts to create the equivalent batched graph.

    Args:
      data_dicts: An iterable of data dictionaries with keys a subset of
        `GRAPH_DATA_FIELDS`, plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`.
        Every element of `data_dicts` has to contain the same set of keys.
        Moreover, the key `NODES` or `N_NODE` must be present in every element of
        `data_dicts`.

    Returns:
      A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
      representing the concatenated graphs.

    Raises:
      ValueError: If two dictionaries in `data_dicts` have a different set of
        keys.
    """
    # Go from a list of dict to a dict of lists
    dct = collections.defaultdict(lambda: [])
    for data_dict in data_dicts:
        data_dict = _populate_number_fields(data_dict)
        for k, v in data_dict.items():
            if v is not None:
                dct[k].append(v)
            elif k not in dct:
                dct[k] = None
    dct = dict(dct)

    # Concatenate the graphs.
    for field, tensors in dct.items():
        if tensors is None:
            dct[field] = None
        elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
            dct[field] = torch.stack(tensors)
        else:
            dct[field] = torch.cat(tensors, dim=0)

    # Add offsets to the receiver and sender indices.
    if dct[RECEIVERS] is not None:
        offset = _compute_stacked_offsets(dct[N_NODE], dct[N_EDGE])
        dct[RECEIVERS] += offset.long()
        dct[SENDERS] += offset.long()

    return dct


def data_dicts_to_graphs_tuple(data_dicts):
    """Creates a `graphs.GraphsTuple` containing tensors from data dicts.

       All dictionaries must have exactly the same set of keys with non-`None`
       values associated to them. Moreover, this set of this key must define a valid
       graph (i.e. if the `EDGES` are `None`, the `SENDERS` and `RECEIVERS` must be
       `None`, and `SENDERS` and `RECEIVERS` can only be `None` both at the same
       time). The values associated with a key must be convertible to `Tensor`s,
       for instance python lists, numpy arrays, or Tensorflow `Tensor`s.

       This method may perform a memory copy.

       The `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to
       `np.int32` type.

      Args:
        data_dicts: An iterable of data dictionaries with keys in `ALL_FIELDS`.
        name: (string, optional) A name for the operation.

      Returns:
        A `graphs.GraphTuple` representing the graphs in `data_dicts`.
    """

    data_dicts = [dict(d) for d in data_dicts]
    for key in ALL_FIELDS:
        for data_dict in data_dicts:
            data_dict.setdefault(key, None)

    utils_np._check_valid_sets_of_keys(data_dicts)  # pylint: disable=protected-access

    data_dicts = _to_compatible_data_dicts(data_dicts)
    return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))
