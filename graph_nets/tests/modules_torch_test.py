# Copyright 2018 Wei-Di Chang and Manfred Diaz. All Rights Reserved.
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
# ============================================================================

"""Tests for modules.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import unittest

from absl.testing import parameterized
from graph_nets import blocks_torch
from graph_nets import graphs
from graph_nets import modules_torch
from graph_nets import utils_np
from graph_nets import utils_torch
import numpy as np
import torch
import torch.nn as nn
import torch_scatter


SMALL_GRAPH_1 = {
    "globals": [1.1, 1.2, 1.3],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [0, 1],
    "receivers": [1, 2],
}

SMALL_GRAPH_2 = {
    "globals": [-1.1, -1.2, -1.3],
    "nodes": [[-10.1, -10.2], [-20.1, -20.2], [-30.1, -30.2]],
    "edges": [[-101., -102., -103., -104.]],
    "senders": [1,],
    "receivers": [2,],
}

SMALL_GRAPH_3 = {
    "globals": [1.1, 1.2, 1.3],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [1, 1],
    "receivers": [0, 2],
}

SMALL_GRAPH_4 = {
    "globals": [1.1, 1.2, 1.3],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [0, 2],
    "receivers": [1, 1],
}


class MLP(nn.Module):
    def __init__(self, output_sizes):
        super(MLP, self).__init__()
        self.output_sizes = output_sizes
        self.mlp = None

    def forward(self, x):
        if self.mlp is None:
            layers = [nn.Linear(x.shape[1], self.output_sizes[0])]
            for i in range(1, len(self.output_sizes)):
                layers.append(nn.Linear(self.output_sizes[i-1], self.output_sizes[i]))
            self.mlp = nn.Sequential(*layers)
        return self.mlp(x)


class Conv2D(nn.Module):
    def __init__(self, output_channels, kernel_shape, stride=1):
        super(Conv2D, self).__init__()
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.conv2d = None


    def forward(self, x):
        if self.conv2d is None:
            self.conv2d = nn.Conv2d(x.shape[1], self.output_channels, self.kernel_shape, stride=self.stride)
        return self.conv2d(x)


class Linear(nn.Module):
    def __init__(self, output_size):
        super(Linear, self).__init__()
        self.output_size = output_size
        self.linear_model = None

    def forward(self, x):
        if self.linear_model is None:
            self.linear_model = nn.Linear(x.shape[-1], self.output_size)

        return self.linear_model(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x



class GraphModuleTest(parameterized.TestCase):
  """Base class for all the tests in this file."""

  def setUp(self):
      super(GraphModuleTest, self).setUp()
      torch.manual_seed(0)

  def _get_input_graph(self, none_fields=None):
      if none_fields is None:
          none_fields = []
      input_graph = utils_torch.data_dicts_to_graphs_tuple(
          [SMALL_GRAPH_1, SMALL_GRAPH_2, SMALL_GRAPH_3, SMALL_GRAPH_4])
      input_graph = input_graph.map(lambda _: None, none_fields)
      return input_graph

  def _get_shaped_input_graph(self):
      return graphs.GraphsTuple(
          nodes=torch.zeros([3, 4, 5, 11], dtype=torch.float32),
          edges=torch.zeros([5, 4, 5, 12], dtype=torch.float32),
          globals=torch.zeros([2, 4, 5, 13], dtype=torch.float32),
          receivers=torch.range(0, 5, dtype=torch.int64) // 3,
          senders=torch.range(0, 5, dtype=torch.int64) % 3,
          n_node=torch.tensor([2, 1], dtype=torch.int64),
          n_edge=torch.tensor([3, 2], dtype=torch.int64),
      )

  def _assert_build_and_run(self, network, input_graph):
      # No error at construction time.
      output = network(input_graph)

  def assertNDArrayNear(self, array1, array2, err):
      self.assertTrue(np.linalg.norm(array1 - array2) < err)

  def assertEqual(self, first, second, msg=...) -> None:
      if isinstance(first, torch.Tensor):
          return self.assertTrue(first.equal(second), msg=msg)

      return super().assertEqual(first, second, msg)

  def assertAllEqual(self, first, second):
      if first is None and second is None:
          return self.assertTrue(first == second)
      return self.assertTrue(torch.all(torch.eq(first, second)))

  def _assert_all_none_or_all_close(self, expected, actual, *args, **kwargs):
    if expected is None:
      return self.assertAllEqual(expected, actual)
    return self.assertAllClose(expected, actual, *args, **kwargs)

  def _get_shaped_model_fns(self):
    edge_model_fn = functools.partial(
        Conv2D, output_channels=10, kernel_shape=[3, 3])
    node_model_fn = functools.partial(
        Conv2D, output_channels=8, kernel_shape=[3, 3])
    global_model_fn = functools.partial(
        Conv2D, output_channels=7, kernel_shape=[3, 3])
    return edge_model_fn, node_model_fn, global_model_fn

  def assertAllClose(self, expected, actual, *args, **kwargs):
      return self.assertTrue(expected.allclose(actual, *args, **kwargs))


class GraphIndependentTest(GraphModuleTest):

  def _get_model(self, name=None):
    kwargs = {
        "edge_model_fn": functools.partial(MLP, output_sizes=[5]),
        "node_model_fn": functools.partial(MLP, output_sizes=[10]),
        "global_model_fn": functools.partial(MLP, output_sizes=[15]),
    }
    if name:
      kwargs["name"] = name
    return modules_torch.GraphIndependent(**kwargs)

  def test_same_as_subblocks(self):
    """Compares the output to explicit subblocks output."""
    input_graph = self._get_input_graph()
    model = self._get_model()
    output_graph = model(input_graph)

    expected_output_edges = model._edge_model(input_graph.edges)
    expected_output_nodes = model._node_model(input_graph.nodes)
    expected_output_globals = model._global_model(input_graph.globals)

    self._assert_all_none_or_all_close(expected_output_edges,
                                       output_graph.edges)
    self._assert_all_none_or_all_close(expected_output_nodes,
                                       output_graph.nodes)
    self._assert_all_none_or_all_close(expected_output_globals,
                                       output_graph.globals)

  @parameterized.named_parameters(
      ("default name", None), ("custom name", "custom_name"))
  def test_created_variables(self, name=None):
    """Verifies variable names and shapes created by a GraphIndependent."""
    name = name if name is not None else "graph_independent"
    expected_var_shapes_dict = {
        name + "/edge_model/mlp/linear_0/b:0": [5],
        name + "/edge_model/mlp/linear_0/w:0": [4, 5],
        name + "/node_model/mlp/linear_0/b:0": [10],
        name + "/node_model/mlp/linear_0/w:0": [2, 10],
        name + "/global_model/mlp/linear_0/b:0": [15],
        name + "/global_model/mlp/linear_0/w:0": [3, 15],
    }
    input_graph = self._get_input_graph()
    model = self._get_model(name=name)

    model(input_graph)
    variables = model.state_dict()
    var_shapes_dict = {var: list(reversed(variables[var].shape)) for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  def test_gradient_flow(self):
    """Verifies that gradient flow is as expected."""
    input_graph = self._get_input_graph()
    model = self._get_model()
    output_graph = model(input_graph)

    for input_field in ["nodes", "edges", "globals"]:
      input_tensor = getattr(input_graph, input_field)
      for output_field in ["nodes", "edges", "globals"]:
        output_tensor = getattr(output_graph, output_field)
        gradients = tf.gradients(output_tensor, input_tensor)
        if input_field == output_field:
          self.assertNotEqual(None, gradients[0])
        else:
          self.assertListEqual([None], gradients)

  @parameterized.named_parameters(
      ("differently shaped edges", "edges"),
      ("differently shaped nodes", "nodes"),
      ("differently shaped globals", "globals"),)
  def test_incompatible_higher_rank_inputs_no_raise(self, field_to_reshape):
    """A GraphIndependent does not make assumptions on its inputs shapes."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, node_model_fn, global_model_fn = self._get_shaped_model_fns()
    input_graph = input_graph.map(
        lambda v: v.permute(0, 2, 1, 3), [field_to_reshape])
    network = modules_torch.GraphIndependent(
        edge_model_fn, node_model_fn, global_model_fn)
    self._assert_build_and_run(network, input_graph)


class GraphNetworkTest(GraphModuleTest):

  def _get_model(self):
    edge_model_fn = functools.partial(Linear, output_size=5)
    node_model_fn = functools.partial(Linear, output_size=10)
    global_model_fn = functools.partial(Linear, output_size=15)
    return modules_torch.GraphNetwork(
        edge_model_fn=edge_model_fn,
        node_model_fn=node_model_fn,
        global_model_fn=global_model_fn)

  @parameterized.named_parameters(
      ("default name", None), ("custom name", "custom_name"))
  def test_created_variables(self, name=None):
    """Verifies variable names and shapes created by a GraphNetwork."""
    name = name if name is not None else "graph_network"
    expected_var_shapes_dict = {
        '_edge_block._edge_model.mlp.0.bias': [5],
        '_edge_block._edge_model.mlp.0.weight': [4 + 4 + 3, 5],
        '_node_block._node_model.mlp.0.bias': [10],
        '_node_block._node_model.mlp.0.weight': [5 + 2 + 3, 10],
        '_global_block._global_model.mlp.0.bias': [15],
        '_global_block._global_model.mlp.0.weight': [10 + 5 + 3, 15],
    }

    input_graph = self._get_input_graph()
    extra_kwargs = {"name": name} if name else {}
    model = modules_torch.GraphNetwork(
        edge_model_fn=functools.partial(MLP, output_sizes=[5]),
        node_model_fn=functools.partial(MLP, output_sizes=[10]),
        global_model_fn=functools.partial(MLP, output_sizes=[15]),
        **extra_kwargs)

    model(input_graph)
    variables = model.state_dict()
    var_shapes_dict = {var: list(reversed(variables[var].shape)) for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("reduce sum reduction", torch_scatter.scatter_add,),
      ("reduce max or zero reduction", blocks_torch.unsorted_segment_max_or_zero,),)
  def test_same_as_subblocks(self, reducer):
    """Compares the output to explicit subblocks output.

    Args:
      reducer: The reducer used in the `NodeBlock` and `GlobalBlock`.
    """
    input_graph = self._get_input_graph()

    edge_model_fn = functools.partial(Linear, output_size=5)
    node_model_fn = functools.partial(Linear, output_size=10)
    global_model_fn = functools.partial(Linear, output_size=15)

    graph_network = modules_torch.GraphNetwork(
        edge_model_fn=edge_model_fn,
        node_model_fn=node_model_fn,
        global_model_fn=global_model_fn,
        reducer=reducer)

    output_graph = graph_network(input_graph)

    edge_block = blocks_torch.EdgeBlock(
        edge_model_fn=lambda: graph_network._edge_block._edge_model,
        use_sender_nodes=True,
        use_edges=True,
        use_receiver_nodes=True,
        use_globals=True)
    node_block = blocks_torch.NodeBlock(
        node_model_fn=lambda: graph_network._node_block._node_model,
        use_nodes=True,
        use_sent_edges=False,
        use_received_edges=True,
        use_globals=True,
        received_edges_reducer=reducer)
    global_block = blocks_torch.GlobalBlock(
        global_model_fn=lambda: graph_network._global_block._global_model,
        use_nodes=True,
        use_edges=True,
        use_globals=True,
        edges_reducer=reducer,
        nodes_reducer=reducer)

    expected_output_edge_block = edge_block(input_graph)
    expected_output_node_block = node_block(expected_output_edge_block)
    expected_output_global_block = global_block(expected_output_node_block)
    expected_edges = expected_output_edge_block.edges
    expected_nodes = expected_output_node_block.nodes
    expected_globals = expected_output_global_block.globals

    self._assert_all_none_or_all_close(expected_edges,
                                       output_graph.edges)
    self._assert_all_none_or_all_close(expected_nodes,
                                       output_graph.nodes)
    self._assert_all_none_or_all_close(expected_globals,
                                       output_graph.globals)

  def test_dynamic_batch_sizes(self):
    """Checks that all batch sizes are as expected through a GraphNetwork."""
    input_graph = self._get_input_graph()
    placeholders = input_graph.map(lambda field: field.unsqueeze(0), graphs.ALL_FIELDS)
    model = self._get_model()
    output = model(placeholders)
    other_input_graph = utils_np.data_dicts_to_graphs_tuple(
          [SMALL_GRAPH_1, SMALL_GRAPH_2])
    for k, v in other_input_graph._asdict().items():
      self.assertEqual(v.shape[0], getattr(output, k).shape[0])

  @parameterized.named_parameters(
      ("float64 data", torch.float64, torch.int64),
      ("int64 indices", torch.float32, torch.int64),)
  def test_dtypes(self, data_dtype, indices_dtype):
      """Checks that all the output types are as expected for blocks."""
      default_type = torch.get_default_dtype()
      torch.set_default_dtype(data_dtype)
      input_graph = self._get_input_graph()
      input_graph = input_graph.map(lambda v: v.type(data_dtype),
                                    ["nodes", "edges", "globals"])
      input_graph = input_graph.map(lambda v: v.type(indices_dtype),
                                    ["receivers", "senders"])
      model = self._get_model()
      output = model(input_graph)
      for field in ["nodes", "globals", "edges"]:
          self.assertEqual(data_dtype, getattr(output, field).dtype)
      for field in ["receivers", "senders"]:
          self.assertEqual(indices_dtype, getattr(output, field).dtype)
      torch.set_default_dtype(default_type)

  @parameterized.named_parameters(
      ("edges only", True, False, False, False),
      ("receivers only", False, True, False, False),
      ("senders only", False, False, True, False),
      ("globals only", False, False, False, True),)
  def test_edge_block_options(self,
                              use_edges,
                              use_receiver_nodes,
                              use_sender_nodes,
                              use_globals):
    """Test for configuring the EdgeBlock options."""
    reducer = torch_scatter.scatter_add
    input_graph = self._get_input_graph()
    edge_model_fn = functools.partial(Linear, output_size=10)
    edge_block_opt = {"use_edges": use_edges,
                      "use_receiver_nodes": use_receiver_nodes,
                      "use_sender_nodes": use_sender_nodes,
                      "use_globals": use_globals}
    # Identity node model
    node_model_fn = lambda: Identity()
    node_block_opt = {"use_received_edges": False,
                      "use_sent_edges": False,
                      "use_nodes": True,
                      "use_globals": False}
    # Identity global model
    global_model_fn = lambda: Identity()
    global_block_opt = {"use_globals": True,
                        "use_nodes": False,
                        "use_edges": False}

    graph_network = modules_torch.GraphNetwork(
        edge_model_fn=edge_model_fn,
        edge_block_opt=edge_block_opt,
        node_model_fn=node_model_fn,
        node_block_opt=node_block_opt,
        global_model_fn=global_model_fn,
        global_block_opt=global_block_opt,
        reducer=reducer)

    output_graph = graph_network(input_graph)

    edge_block = blocks_torch.EdgeBlock(
        edge_model_fn=lambda: graph_network._edge_block._edge_model,
        use_edges=use_edges,
        use_receiver_nodes=use_receiver_nodes,
        use_sender_nodes=use_sender_nodes,
        use_globals=use_globals)

    expected_output_edge_block = edge_block(input_graph)
    expected_output_node_block = expected_output_edge_block
    expected_output_global_block = expected_output_node_block
    expected_edges = expected_output_edge_block.edges
    expected_nodes = expected_output_node_block.nodes
    expected_globals = expected_output_global_block.globals

    self._assert_all_none_or_all_close(expected_edges,
                                       output_graph.edges)
    self._assert_all_none_or_all_close(expected_nodes,
                                       output_graph.nodes)
    self._assert_all_none_or_all_close(expected_globals,
                                       output_graph.globals)

  @parameterized.named_parameters(
      ("received edges only", True, False, False, False, None, None),
      ("received edges, max reduction",
       True, False, False, False, torch_scatter.scatter_add, None),
      ("sent edges only", False, True, False, False, None, None),
      ("sent edges, max reduction",
       False, True, False, False, None, torch_scatter.scatter_add),
      ("nodes only", False, False, True, False, None, None),
      ("globals only", False, False, False, True, None, None),
  )
  def test_node_block_options(self,
                              use_received_edges,
                              use_sent_edges,
                              use_nodes,
                              use_globals,
                              received_edges_reducer,
                              sent_edges_reducer):
    """Test for configuring the NodeBlock options."""
    input_graph = self._get_input_graph()

    if use_received_edges:
      received_edges_reducer = received_edges_reducer or torch_scatter.scatter_add
    if use_sent_edges:
      sent_edges_reducer = sent_edges_reducer or torch_scatter.scatter_add

    # Identity edge model.
    edge_model_fn = lambda: Identity()
    edge_block_opt = {"use_edges": True,
                      "use_receiver_nodes": False,
                      "use_sender_nodes": False,
                      "use_globals": False}
    node_model_fn = functools.partial(Linear, output_size=10)
    node_block_opt = {"use_received_edges": use_received_edges,
                      "use_sent_edges": use_sent_edges,
                      "use_nodes": use_nodes,
                      "use_globals": use_globals,
                      "received_edges_reducer": received_edges_reducer,
                      "sent_edges_reducer": sent_edges_reducer}
    # Identity global model
    global_model_fn = lambda: Identity()
    global_block_opt = {"use_globals": True,
                        "use_nodes": False,
                        "use_edges": False}

    graph_network = modules_torch.GraphNetwork(
        edge_model_fn=edge_model_fn,
        edge_block_opt=edge_block_opt,
        node_model_fn=node_model_fn,
        node_block_opt=node_block_opt,
        global_model_fn=global_model_fn,
        global_block_opt=global_block_opt)

    output_graph = graph_network(input_graph)

    node_block = blocks_torch.NodeBlock(
        node_model_fn=lambda: graph_network._node_block._node_model,
        use_nodes=use_nodes,
        use_sent_edges=use_sent_edges,
        use_received_edges=use_received_edges,
        use_globals=use_globals,
        received_edges_reducer=received_edges_reducer,
        sent_edges_reducer=sent_edges_reducer)

    expected_output_edge_block = input_graph
    expected_output_node_block = node_block(input_graph)
    expected_output_global_block = expected_output_node_block
    expected_edges = expected_output_edge_block.edges
    expected_nodes = expected_output_node_block.nodes
    expected_globals = expected_output_global_block.globals

    self._assert_all_none_or_all_close(expected_edges,
                                       output_graph.edges)
    self._assert_all_none_or_all_close(expected_nodes,
                                       output_graph.nodes)
    self._assert_all_none_or_all_close(expected_globals,
                                       output_graph.globals)

  @parameterized.named_parameters(
      ("edges only", True, False, False, None, None),
      ("edges only, max", True, False, False, torch_scatter.scatter_max, None),
      ("nodes only", False, True, False, None, None),
      ("nodes only, max", False, True, False, None, torch_scatter.scatter_max),
      ("globals only", False, False, True, None, None),
  )
  def test_global_block_options(self,
                                use_edges,
                                use_nodes,
                                use_globals,
                                edges_reducer,
                                nodes_reducer):
    """Test for configuring the NodeBlock options."""
    input_graph = self._get_input_graph()

    if use_edges:
      edges_reducer = edges_reducer or torch_scatter.scatter_add
    if use_nodes:
      nodes_reducer = nodes_reducer or torch_scatter.scatter_add

    # Identity edge model.
    edge_model_fn = lambda: Identity()
    edge_block_opt = {"use_edges": True,
                      "use_receiver_nodes": False,
                      "use_sender_nodes": False,
                      "use_globals": False}
    # Identity node model
    node_model_fn = lambda: Identity()
    node_block_opt = {"use_received_edges": False,
                      "use_sent_edges": False,
                      "use_nodes": True,
                      "use_globals": False}
    global_model_fn = functools.partial(Linear, output_size=10)
    global_block_opt = {"use_globals": use_globals,
                        "use_nodes": use_nodes,
                        "use_edges": use_edges,
                        "edges_reducer": edges_reducer,
                        "nodes_reducer": nodes_reducer}

    graph_network = modules_torch.GraphNetwork(
        edge_model_fn=edge_model_fn,
        edge_block_opt=edge_block_opt,
        node_model_fn=node_model_fn,
        node_block_opt=node_block_opt,
        global_model_fn=global_model_fn,
        global_block_opt=global_block_opt)

    output_graph = graph_network(input_graph)

    global_block = blocks_torch.GlobalBlock(
        global_model_fn=lambda: graph_network._global_block._global_model,
        use_edges=use_edges,
        use_nodes=use_nodes,
        use_globals=use_globals,
        edges_reducer=edges_reducer,
        nodes_reducer=nodes_reducer)

    expected_output_edge_block = input_graph
    expected_output_node_block = expected_output_edge_block
    expected_output_global_block = global_block(expected_output_node_block)
    expected_edges = expected_output_edge_block.edges
    expected_nodes = expected_output_node_block.nodes
    expected_globals = expected_output_global_block.globals

    self._assert_all_none_or_all_close(expected_edges,
                                       output_graph.edges)
    self._assert_all_none_or_all_close(expected_nodes,
                                       output_graph.nodes)
    self._assert_all_none_or_all_close(expected_globals,
                                       output_graph.globals)

  def test_higher_rank_outputs(self):
    """Tests that a graph net can be build with higher rank inputs/outputs."""
    input_graph = self._get_shaped_input_graph()
    network = modules_torch.GraphNetwork(*self._get_shaped_model_fns())
    self._assert_build_and_run(network, input_graph)

  @parameterized.named_parameters(
      ("wrongly shaped edges", "edges"),
      ("wrongly shaped nodes", "nodes"),
      ("wrongly shaped globals", "globals"),)
  def test_incompatible_higher_rank_inputs_raises(self, field_to_reshape):
    """A exception should be raised if the inputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, node_model_fn, global_model_fn = self._get_shaped_model_fns()
    input_graph = input_graph.map(
        lambda v: v.permute(0, 2, 1, 3), [field_to_reshape])
    graph_network = modules_torch.GraphNetwork(
        edge_model_fn, node_model_fn, global_model_fn)
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      graph_network(input_graph)

  def test_incompatible_higher_rank_partial_outputs_raises(self):
    """A error should be raised if partial outputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, node_model_fn, global_model_fn = self._get_shaped_model_fns()
    edge_model_fn_2 = functools.partial(
        Conv2D, output_channels=10, kernel_shape=[3, 3], stride=[1, 2])
    graph_network = modules_torch.GraphNetwork(
        edge_model_fn_2, node_model_fn, global_model_fn)
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      graph_network(input_graph)
    node_model_fn_2 = functools.partial(
        Conv2D, output_channels=10, kernel_shape=[3, 3], stride=[1, 2])
    graph_network = modules_torch.GraphNetwork(
        edge_model_fn, node_model_fn_2, global_model_fn)
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      graph_network(input_graph)


class InteractionNetworkTest(GraphModuleTest):

  def _get_model(self, reducer=None, name=None):
    kwargs = {
        "edge_model_fn": functools.partial(Linear, output_size=5),
        "node_model_fn": functools.partial(Linear, output_size=10)
    }
    if reducer:
      kwargs["reducer"] = reducer
    if name:
      kwargs["name"] = name
    return modules_torch.InteractionNetwork(**kwargs)

  @parameterized.named_parameters(
      ("default name", None), ("custom name", "custom_name"))
  def test_created_variables(self, name=None):
    """Verifies variable names and shapes created by an InteractionNetwork."""
    name = name if name is not None else "interaction_network"
    expected_var_shapes_dict = {
        '_edge_block._edge_model.linear_model.bias': [5],
        '_edge_block._edge_model.linear_model.weight': [8, 5],
        '_node_block._node_model.linear_model.bias': [10],
        '_node_block._node_model.linear_model.weight': [7, 10],
    }
    input_graph = utils_torch.data_dicts_to_graphs_tuple([SMALL_GRAPH_1])
    model = self._get_model(name=name)

    model(input_graph)
    variables = model.state_dict()
    var_shapes_dict = {var: list(reversed(variables[var].shape)) for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("default", torch_scatter.scatter_add,),
      ("max or zero reduction", blocks_torch.unsorted_segment_max_or_zero,),
      ("no globals", torch_scatter.scatter_add, "globals"),
  )
  def test_same_as_subblocks(self, reducer, none_field=None):
    """Compares the output to explicit subblocks output.

    Args:
      reducer: The reducer used in the `NodeBlock`s.
      none_field: (string, default=None) If not None, the corresponding field
        is removed from the input graph.
    """
    input_graph = self._get_input_graph(none_field)

    interaction_network = self._get_model(reducer)
    output_graph = interaction_network(input_graph)
    edges_out = output_graph.edges
    nodes_out = output_graph.nodes
    self.assertAllEqual(input_graph.globals, output_graph.globals)

    edge_block = blocks_torch.EdgeBlock(
        edge_model_fn=lambda: interaction_network._edge_block._edge_model,
        use_sender_nodes=True,
        use_edges=True,
        use_receiver_nodes=True,
        use_globals=False)
    node_block = blocks_torch.NodeBlock(
        node_model_fn=lambda: interaction_network._node_block._node_model,
        use_nodes=True,
        use_sent_edges=False,
        use_received_edges=True,
        use_globals=False,
        received_edges_reducer=reducer)

    expected_output_edge_block = edge_block(input_graph)
    expected_output_node_block = node_block(expected_output_edge_block)
    expected_edges = expected_output_edge_block.edges
    expected_nodes = expected_output_node_block.nodes

    self._assert_all_none_or_all_close(expected_edges, edges_out)
    self._assert_all_none_or_all_close(expected_nodes, nodes_out)

  @parameterized.named_parameters(
      ("no nodes", ["nodes"],),
      ("no edge data", ["edges"],),
      ("no edges", ["edges", "receivers", "senders"],),
  )
  def test_field_must_not_be_none(self, none_fields):
    """Tests that the model cannot be built if required fields are missing."""
    input_graph = utils_torch.data_dicts_to_graphs_tuple([SMALL_GRAPH_1])
    input_graph = input_graph.map(lambda _: None, none_fields)
    interaction_network = self._get_model()
    with self.assertRaises(ValueError):
      interaction_network(input_graph)

  def test_higher_rank_outputs(self):
    """Tests that an IN can be build with higher rank inputs/outputs."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, node_model_fn, _ = self._get_shaped_model_fns()
    graph_network = modules_torch.InteractionNetwork(edge_model_fn, node_model_fn)
    self._assert_build_and_run(graph_network, input_graph)

  @parameterized.named_parameters(
      ("wrongly shaped edges", "edges"),
      ("wrongly shaped nodes", "nodes"),)
  def test_incompatible_higher_rank_inputs_raises(self, field_to_reshape):
    """Am exception should be raised if the inputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, node_model_fn, _ = self._get_shaped_model_fns()
    input_graph = input_graph.map(
        lambda v: v.permute(0, 2, 1, 3), [field_to_reshape])
    graph_network = modules_torch.InteractionNetwork(edge_model_fn, node_model_fn)
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      graph_network(input_graph)

  def test_incompatible_higher_rank_inputs_no_raise(self):
    """The globals can have an arbitrary shape in the input."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, node_model_fn, _ = self._get_shaped_model_fns()
    input_graph = input_graph.replace(
        globals=input_graph.globals.permute(0, 2, 1, 3))
    graph_network = modules_torch.InteractionNetwork(edge_model_fn, node_model_fn)
    self._assert_build_and_run(graph_network, input_graph)


class RelationNetworkTest(GraphModuleTest):

  def _get_model(self, reducer=torch_scatter.scatter_add, name=None):
    kwargs = {
        "edge_model_fn": functools.partial(Linear, output_size=5),
        "global_model_fn": functools.partial(Linear, output_size=15)
    }
    if reducer:
      kwargs["reducer"] = reducer
    if name:
      kwargs["name"] = name
    return modules_torch.RelationNetwork(**kwargs)

  @parameterized.named_parameters(
      ("default name", None), ("custom name", "custom_name"))
  def test_created_variables(self, name=None):
    """Verifies variable names and shapes created by a RelationNetwork."""
    name = name if name is not None else "relation_network"
    expected_var_shapes_dict = {
        '_edge_block._edge_model.linear_model.bias': [5],
        '_edge_block._edge_model.linear_model.weight': [4, 5],
        '_global_block._global_model.linear_model.bias': [15],
        '_global_block._global_model.linear_model.weight': [5, 15],
    }
    input_graph = utils_torch.data_dicts_to_graphs_tuple([SMALL_GRAPH_1])
    model = self._get_model(name=name)

    model(input_graph)
    variables = model.state_dict()
    var_shapes_dict = {var: list(reversed(variables[var].shape)) for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("default", torch_scatter.scatter_add, None),
      ("max or zero reduction", blocks_torch.unsorted_segment_max_or_zero, None),
      ("no edges", torch_scatter.scatter_add, "edges"),
      ("no globals", torch_scatter.scatter_add, "globals"),
  )
  def test_same_as_subblocks(self, reducer, none_field=None):
    """Compares the output to explicit subblocks output.

    Args:
      reducer: The reducer used in the `GlobalBlock`.
      none_field: (string, default=None) If not None, the corresponding field
        is removed from the input graph.
    """
    input_graph = self._get_input_graph(none_field)
    relation_network = self._get_model(reducer)
    output_graph = relation_network(input_graph)

    edge_block = blocks_torch.EdgeBlock(
        edge_model_fn=lambda: relation_network._edge_block._edge_model,
        use_edges=False,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=False)
    global_block = blocks_torch.GlobalBlock(
        global_model_fn=lambda: relation_network._global_block._global_model,
        use_edges=True,
        use_nodes=False,
        use_globals=False,
        edges_reducer=reducer,
        nodes_reducer=reducer)

    expected_output_edge_block = edge_block(input_graph)
    expected_output_global_block = global_block(expected_output_edge_block)

    self.assertEqual(input_graph.edges, output_graph.edges)
    self.assertEqual(input_graph.nodes, output_graph.nodes)

    self._assert_all_none_or_all_close(output_graph.globals, expected_output_global_block.globals)

  @parameterized.named_parameters(
      ("no nodes", ["nodes"],), ("no edges", ["edges", "receivers", "senders"],)
  )
  def test_field_must_not_be_none(self, none_fields):
    """Tests that the model cannot be built if required fields are missing."""
    input_graph = utils_torch.data_dicts_to_graphs_tuple([SMALL_GRAPH_1])
    input_graph = input_graph.map(lambda _: None, none_fields)
    relation_network = self._get_model()
    with self.assertRaises(ValueError):
      relation_network(input_graph)

  @parameterized.named_parameters(
      ("differently shaped edges", "edges"),
      ("differently shaped nodes", "nodes"),
      ("differently shaped globals", "globals"),)
  def test_incompatible_higher_rank_inputs_no_raise(self, field_to_reshape):
    """A RelationNetwork does not make assumptions on its inputs shapes."""
    input_graph = self._get_shaped_input_graph()
    edge_model_fn, _, global_model_fn = self._get_shaped_model_fns()
    input_graph = input_graph.map(
        lambda v: v.permute(0, 2, 1, 3), [field_to_reshape])
    network = modules_torch.RelationNetwork(edge_model_fn, global_model_fn)
    self._assert_build_and_run(network, input_graph)


class DeepSetsTest(GraphModuleTest):

  def _get_model(self, reducer=None, name=None):
    kwargs = {
        "node_model_fn": functools.partial(Linear, output_size=5),
        "global_model_fn": functools.partial(Linear, output_size=15)
    }
    if reducer:
      kwargs["reducer"] = reducer
    if name:
      kwargs["name"] = name
    return modules_torch.DeepSets(**kwargs)

  @parameterized.named_parameters(
      ("default name", None), ("custom name", "custom_name"))
  def test_created_variables(self, name=None):
    """Verifies variable names and shapes created by a DeepSets network."""
    name = name if name is not None else "deep_sets"
    expected_var_shapes_dict = {
        '_global_block._global_model.linear_model.bias': [15],
        '_global_block._global_model.linear_model.weight': [5, 15],
        '_node_block._node_model.linear_model.bias': [5],
        '_node_block._node_model.linear_model.weight': [5, 5],
    }
    input_graph = self._get_input_graph()
    model = self._get_model(name=name)

    model(input_graph)
    variables = model.state_dict()
    var_shapes_dict = {var: list(reversed(variables[var].shape)) for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("default", torch_scatter.scatter_add, []),
      ("no edge data", torch_scatter.scatter_add, ["edges"]),
      ("no edges", torch_scatter.scatter_add, ["edges", "receivers", "senders"]),
      ("max or zero reduction", blocks_torch.unsorted_segment_max_or_zero, []),
  )
  def test_same_as_subblocks(self, reducer, none_fields):
    """Compares the output to explicit subblocks output.

    Args:
      reducer: The reducer used in the NodeBlock.
      none_fields: (list of strings) The corresponding fields are removed from
        the input graph.
    """
    input_graph = self._get_input_graph()
    input_graph = input_graph.map(lambda _: None, none_fields)

    deep_sets = self._get_model(reducer)

    output_graph = deep_sets(input_graph)
    output_nodes = output_graph.nodes
    output_globals = output_graph.globals

    node_block = blocks_torch.NodeBlock(
        node_model_fn=lambda: deep_sets._node_block._node_model,
        use_received_edges=False,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=True)
    global_block = blocks_torch.GlobalBlock(
        global_model_fn=lambda: deep_sets._global_block._global_model,
        use_edges=False,
        use_nodes=True,
        use_globals=False,
        nodes_reducer=reducer)

    node_block_out = node_block(input_graph)
    expected_nodes = node_block_out.nodes
    expected_globals = global_block(node_block_out).globals

    self.assertAllEqual(input_graph.edges, output_graph.edges)
    self.assertAllEqual(input_graph.receivers, output_graph.receivers)
    self.assertAllEqual(input_graph.senders, output_graph.senders)

    self._assert_all_none_or_all_close(expected_nodes, output_nodes)
    self._assert_all_none_or_all_close(expected_globals, output_globals)

  @parameterized.parameters(
      ("nodes",), ("globals",),
  )
  def test_field_must_not_be_none(self, none_field):
    """Tests that the model cannot be built if required fields are missing."""
    input_graph = utils_torch.data_dicts_to_graphs_tuple([SMALL_GRAPH_1])
    input_graph = input_graph.replace(**{none_field: None})
    deep_sets = self._get_model()
    with self.assertRaises(ValueError):
      deep_sets(input_graph)

  def test_incompatible_higher_rank_inputs_raises(self):
    """A exception should be raised if the inputs have incompatible shapes."""
    input_graph = self._get_shaped_input_graph()
    _, node_model_fn, global_model_fn = self._get_shaped_model_fns()
    input_graph = input_graph.replace(
        nodes=input_graph.nodes.permute(0, 2, 1, 3))
    graph_network = modules_torch.DeepSets(node_model_fn, global_model_fn)
    with self.assertRaisesRegexp(ValueError, "in both shapes must be equal"):
      graph_network(input_graph)

  def test_incompatible_higher_rank_partial_outputs_no_raise(self):
    """There is no constraint on the size of the partial outputs."""
    input_graph = self._get_shaped_input_graph()
    node_model_fn = functools.partial(
        Conv2D, output_channels=10, kernel_shape=[3, 3], stride=[1, 2])
    global_model_fn = functools.partial(
        Conv2D, output_channels=10, kernel_shape=[3, 3])
    network = modules_torch.DeepSets(node_model_fn, global_model_fn)
    self._assert_build_and_run(network, input_graph)

  def test_incompatible_higher_rank_inputs_no_raise(self):
    """A DeepSets does not make assumptions on the shape if its input edges."""
    input_graph = self._get_shaped_input_graph()
    _, node_model_fn, global_model_fn = self._get_shaped_model_fns()
    input_graph = input_graph.replace(
        edges=input_graph.edges.permute(0, 2, 1, 3))
    network = modules_torch.DeepSets(node_model_fn, global_model_fn)
    self._assert_build_and_run(network, input_graph)


class CommNetTest(GraphModuleTest):

  def _get_model(self, reducer=None, name=None):
    kwargs = {
        "edge_model_fn": functools.partial(Linear, output_size=15),
        "node_encoder_model_fn": functools.partial(Linear, output_size=8),
        "node_model_fn": functools.partial(Linear, output_size=5),
    }
    if reducer is not None:
      kwargs["reducer"] = reducer
    if name:
      kwargs["name"] = name
    return modules_torch.CommNet(**kwargs)

  @parameterized.named_parameters(
      ("default name", None), ("custom name", "custom_name"))
  def test_created_variables(self, name=None):
    """Verifies variable names and shapes created by a DeepSets network."""
    name = name if name is not None else "comm_net"
    expected_var_shapes_dict = {
        '_edge_block._edge_model.linear_model.bias': [15],
        '_edge_block._edge_model.linear_model.weight': [2, 15],
        '_node_block._node_model.linear_model.bias': [5],
        '_node_block._node_model.linear_model.weight': [23, 5],
        '_node_encoder_block._node_model.linear_model.bias': [8],
        '_node_encoder_block._node_model.linear_model.weight': [2, 8]
    }
    input_graph = self._get_input_graph()
    model = self._get_model(name=name)

    model(input_graph)
    variables = model.state_dict()
    var_shapes_dict = {var: list(reversed(variables[var].shape)) for var in variables}
    self.assertDictEqual(expected_var_shapes_dict, var_shapes_dict)

  @parameterized.named_parameters(
      ("default", torch_scatter.scatter_add,),
      ("no edges", torch_scatter.scatter_add, "edges"),
      ("no globals", torch_scatter.scatter_add, "globals"),
      ("max or zero reduction", blocks_torch.unsorted_segment_max_or_zero,),
  )
  def test_same_as_subblocks(self, reducer, none_field=None):
    """Compares the output to explicit subblocks output.

    Args:
      reducer: The reducer used in the `NodeBlock`s.
      none_field: (string, default=None) If not None, the corresponding field
        is removed from the input graph.
    """
    input_graph = self._get_input_graph(none_field)

    comm_net = self._get_model(reducer)
    output_graph = comm_net(input_graph)
    output_nodes = output_graph.nodes

    edge_subblock = blocks_torch.EdgeBlock(
        edge_model_fn=lambda: comm_net._edge_block._edge_model,
        use_edges=False,
        use_receiver_nodes=False,
        use_sender_nodes=True,
        use_globals=False)
    node_encoder_subblock = blocks_torch.NodeBlock(
        node_model_fn=lambda: comm_net._node_encoder_block._node_model,
        use_received_edges=False,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=False,
        received_edges_reducer=reducer)
    node_subblock = blocks_torch.NodeBlock(
        node_model_fn=lambda: comm_net._node_block._node_model,
        use_received_edges=True,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=False,
        received_edges_reducer=reducer)

    edge_block_out = edge_subblock(input_graph)
    encoded_nodes = node_encoder_subblock(input_graph).nodes
    node_input_graph = input_graph.replace(
        edges=edge_block_out.edges, nodes=encoded_nodes)
    node_block_out = node_subblock(node_input_graph)
    expected_nodes = node_block_out.nodes

    self.assertAllEqual(input_graph.globals, output_graph.globals)
    self.assertAllEqual(input_graph.edges, output_graph.edges)
    self.assertAllEqual(input_graph.receivers, output_graph.receivers,)
    self.assertAllEqual(input_graph.senders, output_graph.senders)

    self._assert_all_none_or_all_close(expected_nodes, output_nodes)

  @parameterized.named_parameters(
      ("no nodes", ["nodes"],), ("no edges", ["edges", "receivers", "senders"],)
  )
  def test_field_must_not_be_none(self, none_fields):
    """Tests that the model cannot be built if required fields are missing."""
    input_graph = utils_torch.data_dicts_to_graphs_tuple([SMALL_GRAPH_1])
    input_graph = input_graph.map(lambda _: None, none_fields)
    comm_net = self._get_model()
    with self.assertRaises(ValueError):
      comm_net(input_graph)

  def test_higher_rank_outputs(self):
    """Tests that a CommNet can be build with higher rank inputs/outputs."""
    input_graph = self._get_shaped_input_graph()
    graph_network = modules_torch.CommNet(*self._get_shaped_model_fns())
    self._assert_build_and_run(graph_network, input_graph)


class SelfAttentionTest(GraphModuleTest):

  def _get_model(self, reducer=None, name=None):
    kwargs = {
        "edge_model_fn": functools.partial(Linear, output_size=15),
        "node_encoder_model_fn": functools.partial(Linear, output_size=8),
        "node_model_fn": functools.partial(Linear, output_size=5),
    }
    if reducer is not None:
      kwargs["reducer"] = reducer
    if name:
      kwargs["name"] = name
    return modules_torch.CommNet(**kwargs)

  LOGITS_1D = torch.tensor([np.log(2), np.log(2), np.log(2), 0., 0., 0.])
  SOFTMAX_1D = torch.tensor([1., 2/3, 0.5, 0.25, 0.25, 1/3])
  LOGITS_2D = torch.tensor([[np.log(2), 1.], [np.log(2), 1.], [np.log(2), 1.],
               [0., 1.], [0., 1.], [0., 1.]])
  SOFTMAX_2D = torch.tensor([[1., 1.], [2/3, 0.5], [1/2, 1/3],
                [1/4, 1/3], [1/4, 1/3], [1/3, 0.5]])
  SENDERS = [0, 2, 2, 3, 4, 3]
  RECEIVERS = [1, 5, 6, 6, 6, 5]
  N_NODE = [2, 5]
  N_EDGE = [1, 5]

  @parameterized.named_parameters(
      ("one dimensional", LOGITS_1D, SOFTMAX_1D),
      ("two dimensional", LOGITS_2D, SOFTMAX_2D),)
  def test_unsorted_segment_softmax(self, data, expected_softmax):
    """Verifies variable names and shapes created by a DeepSets network."""

    data = torch.tensor(data, dtype=torch.float32)
    segment_ids = torch.tensor(self.RECEIVERS, dtype=torch.int64)
    num_segments = torch.tensor(sum(self.N_NODE), dtype=torch.int64)

    actual_softmax = modules_torch._unsorted_segment_softmax(
        data, segment_ids, num_segments)

    self.assertAllClose(expected_softmax, actual_softmax)

  @parameterized.named_parameters(
      ("one dimensional", LOGITS_1D, SOFTMAX_1D,
       modules_torch._unsorted_segment_softmax),
      ("two dimensional", LOGITS_2D, SOFTMAX_2D,
       modules_torch._unsorted_segment_softmax),)
  def test_received_edges_normalizer(self, logits,
                                     expected_normalized, normalizer):
    graph = graphs.GraphsTuple(
        nodes=None,
        edges=torch.tensor(logits, dtype=torch.float32),
        globals=None,
        receivers=torch.tensor(self.RECEIVERS, dtype=torch.int64),
        senders=torch.tensor(self.SENDERS, dtype=torch.int64),
        n_node=torch.tensor(self.N_NODE, dtype=torch.int64),
        n_edge=torch.tensor(self.N_EDGE, dtype=torch.int64),
    )
    actual_normalized_edges = modules_torch._received_edges_normalizer(
        graph, normalizer)
    self.assertAllClose(expected_normalized, actual_normalized_edges)

  def test_self_attention(self):
    # Just one feature per node.
    values_np = np.arange(sum(self.N_NODE)) + 1.
    # Multiple heads, one positive values, one negative values.
    values_np = np.stack([values_np, values_np*-1.], axis=-1)
    # Multiple features per node, per head, at different scales.
    values_np = np.stack([values_np, values_np*0.1], axis=-1)
    values = torch.tensor(values_np, dtype=torch.float32)

    keys_np = [
        [[0.3, 0.4]]*2,  # Irrelevant (only sender to one node)
        [[0.1, 0.5]]*2,  # Not used (is not a sender)
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 1], [1, 1]],
        [[0.4, 0.3]]*2,  # Not used (is not a sender)
        [[0.3, 0.2]]*2]  # Not used (is not a sender)
    keys = torch.tensor(keys_np, dtype=torch.float32)

    queries_np = [
        [[0.2, 0.7]]*2,  # Not used (is not a receiver)
        [[0.3, 0.2]]*2,  # Irrelevant (only receives from one node)
        [[0.2, 0.8]]*2,  # Not used (is not a receiver)
        [[0.2, 0.4]]*2,  # Not used (is not a receiver)
        [[0.3, 0.9]]*2,  # Not used (is not a receiver)
        [[0, np.log(2)], [np.log(3), 0]],
        [[np.log(2), 0], [0, np.log(3)]]]
    queries = torch.tensor(queries_np, dtype=torch.float32)

    attention_graph = graphs.GraphsTuple(
        nodes=None,
        edges=None,
        globals=None,
        receivers=torch.tensor(self.RECEIVERS, dtype=torch.int64),
        senders=torch.tensor(self.SENDERS, dtype=torch.int64),
        n_node=torch.tensor(self.N_NODE, dtype=torch.int64),
        n_edge=torch.tensor(self.N_EDGE, dtype=torch.int64)
    )

    self_attention = modules_torch.SelfAttention()
    output_graph = self_attention(values, keys, queries, attention_graph)
    mixed_nodes = output_graph.nodes


    expected_mixed_nodes = torch.tensor([
        [[0., 0.], [0., 0.]],  # Does not receive any edges
        [[1., 0.1], [-1., -0.1]],  # Only receives from n0.
        [[0., 0.], [0., 0.]],  # Does not receive any edges
        [[0., 0.], [0., 0.]],  # Does not receive any edges
        [[0., 0.], [0., 0.]],  # Does not receive any edges
        [[11/3, 11/3*0.1],  # Head one, receives from n2(1/3) n3(2/3)
         [-15/4, -15/4*0.1]],  # Head two, receives from n2(1/4) n3(3/4)
        [[20/5, 20/5*0.1],   # Head one, receives from n2(2/5) n3(1/5) n4(2/5)
         [-28/7, -28/7*0.1]],  # Head two, receives from n2(3/7) n3(1/7) n4(3/7)
    ])

    self.assertAllClose(expected_mixed_nodes, mixed_nodes)

if __name__ == "__main__":
    unittest.main()
