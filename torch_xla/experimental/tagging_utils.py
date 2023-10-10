import copy
import dataclasses
from dataclasses import dataclass
import json
import torch
from torch.fx import subgraph_rewriter
from torch.fx import Graph, GraphModule
import torch_xla
from torch_xla.core import xla_model as xm
from typing import List, Tuple, Dict, Any, Callable, Union, Optional

__all__ = ["mark_pattern"]


@dataclass
class PortTag:
    name: str  # Identify Patttern
    pos: int  # Arg/return position
    id: int  # Patten instance id
    is_input: bool = True  # If the tagged tensor is input/output
    attr: Dict = None  # Attribute of the pattern, only output has attr field


class TagSerializer(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def tag_input(x, i, tag_name, total_input):
    if tag_name not in tag_input.counter:
        tag_input.counter[tag_name] = 0
    tag_count = tag_input.counter[tag_name]
    match_id = int(tag_count / total_input)
    print(
        "tag_input name: {}, input pos: {}, match_id: {}".format(tag_name, i, match_id)
    )
    torch_xla._XLAC._xla_add_tag(
        x, json.dumps(PortTag(tag_name, i, match_id, is_input=True), cls=TagSerializer)
    )
    tag_input.counter[tag_name] += 1
    return x


tag_input.counter = dict()


def select_output(outputs, pos):
    return outputs[pos]


def tag_output(x, pos, tag_name, total_output, kwargs):
    if tag_name not in tag_output.counter:
        tag_output.counter[tag_name] = 0
    tag_count = tag_output.counter[tag_name]
    match_id = int(tag_count / total_output)
    print(
        "tag_output name: {}, output pos {}, match_id: {}, attr: {}".format(
            tag_name, pos, match_id, kwargs
        )
    )
    torch_xla._XLAC._xla_add_tag(
        x,
        json.dumps(
            PortTag(tag_name, pos, match_id, is_input=False, attr=kwargs),
            cls=TagSerializer,
        ),
    )
    tag_output.counter[tag_name] += 1
    return x


tag_output.counter = dict()


def get_pattern_node(pattern_name, pattern, args, kwargs):
    pattern_ep = torch.export.export(pattern, args, kwargs)
    n_inputs = len(pattern_ep.graph_signature.user_inputs)
    n_outputs = len(pattern_ep.graph_signature.user_outputs)
    print("pattern has {} inputs, {} outputs.".format(n_inputs, n_outputs))

    new_g = Graph()
    placeholders = []
    # for i in range(n_inputs):
    # FIXME: try kwargs contain tensor
    # Skip constant in args and kwargs
    # currently assume constant is all in kwargs, args only contain tensors
    n_fx_input = len(args)
    for i in range(n_fx_input):
        placeholders.append(new_g.placeholder("input_{}".format(i)))

    tagged_placeholders = []
    # for i in range(n_inputs):
    for i in range(n_fx_input):
        tagged_placeholders.append(
            new_g.call_function(
                # tag_input, (placeholders[i], i, pattern_name, n_inputs)
                tag_input,
                (placeholders[i], i, pattern_name, n_fx_input)
                # tag_input, (placeholders[i], i, "pattern", n_inputs)
            )
        )

    if isinstance(pattern, torch.nn.Module):
        node_tagged = new_g.call_module("pattern")
    else:
        node_tagged = new_g.call_function(pattern, tuple(tagged_placeholders), kwargs)

    output_nodes = []
    if n_outputs > 1:
        for pos in range(n_outputs):
            output_nodes.append(new_g.call_function(select_output, (node_tagged, pos)))
    else:
        output_nodes = [node_tagged]

    tagged_output_nodes = []
    for pos, output in enumerate(output_nodes):
        node_tagged_out = new_g.call_function(
            tag_output, (output, pos, pattern_name, n_outputs, kwargs)
        )
        tagged_output_nodes.append(node_tagged_out)

    node_out = new_g.output(tuple(tagged_output_nodes))
    replace_gm = GraphModule(dict(), new_g)
    return replace_gm


@dataclass
class NodeConstantLoc:
    arg_name: str
    node_name: str
    pos: int = None
    key: str = None


def extract_constant_from_matched_pattern(
    matches: List[subgraph_rewriter.ReplacedPatterns], loc: NodeConstantLoc
):
    val = None
    for match in matches:
        for k, v in match.nodes_map.items():
            if k.name == loc.node_name:
                # print(str(v.args[loc.pos]))
                if loc.pos is not None:
                    val = str(v.args[loc.pos])
                # TODO Handel kwarg
        assert val is not None
        for n in match.replacements:
            if n.op == "call_function" and n.target == tag_output:
                attr_arg_idx = 4  # TODO: move to kwarg of the 'tag_ouptut'
                attr_dict = dict(n.args[attr_arg_idx])
                attr_dict[loc.arg_name] = val
                n.update_arg(4, attr_dict)


def find_constant_arg_mapping(pattern, argsList: List[Tuple], kwargsList:List[Dict]):
    # Assume const in kwargs
    print("in find_constant_arg_mapping")
    assert len(argsList) == 2 and len(kwargsList) == 2 
    # pos = -1
    # for idx in range(len(argsList[0])):
    #     arg0 = argsList[0][idx]
    #     arg1 = argsList[1][idx]
    #     assert type(arg0) == type(arg1)
    #     print(type(arg0))
    #     if not isinstance(arg0, torch.Tensor):
    #         # Only caputure the first constant arg
    #         pos = idx
    #         break
    # if pos == -1:
    #     print("Constant not found")
    #     return None

    ep1 = torch.export.export(pattern, argsList[0], kwargs=kwargsList[0])
    ep2 = torch.export.export(pattern, argsList[1], kwargs=kwargsList[1])

    node_list1 = list(ep1.graph_module.graph.nodes)
    node_list2 = list(ep2.graph_module.graph.nodes)
    assert len(node_list1) == len(node_list2)
    # TODO: Extensive check on topological order and op type
    constant_key = list(kwargsList[0].keys())[0]
    constant_type = type(kwargsList[0][constant_key])
    print(constant_type)
    for idx in range(len(node_list1)):
        n1_args = node_list1[idx].args
        n2_args = node_list2[idx].args
        if len(n1_args) > 0:
            for arg_idx in range(len(n1_args)):
                n1_val = n1_args[arg_idx]
                n2_val = n2_args[arg_idx]
                print(type(n1_val))
                if type(n1_val) == constant_type and n1_val != n2_val:
                    return NodeConstantLoc(constant_key, node_list1[idx].name, pos=arg_idx)
    return None

def eliminate_dangling_arg(graph: Graph):
    nodes_to_erase = []
    for n in graph.nodes:
        if n.op == "placeholder" and len(n.users) == 0:
            nodes_to_erase.append(n)
    for n in nodes_to_erase:
        graph.erase_node(n)

def mark_pattern(
    pattern_name: str,
    exported_ep: GraphModule,
    pattern: Union[Callable, GraphModule, torch.nn.Module],
    pattern_args: Union[Tuple, List[Tuple]],
    pattern_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    constant_fx_node_name: Optional[NodeConstantLoc] = None,
):
    print("check whole graph")
    exported_ep.graph_module.graph.print_tabular()
    pattern_args_for_export = pattern_args
    if not isinstance(pattern_args, Tuple):
        pattern_args_for_export = pattern_args[0]
    pattern_kwargs_for_export = pattern_kwargs or {}
    if not isinstance(pattern_kwargs, Dict) and pattern_kwargs is not None:
        pattern_kwargs_for_export = pattern_kwargs[0]

    if isinstance(pattern, GraphModule):
        pattern_ep = pattern
    else:
        # pattern_ep = torch.export.export(pattern, pattern_args, pattern_kwargs)
        # FIXME: torch.export will generate a dangling input if there is constant
        pattern_ep = torch.export.export(pattern, pattern_args_for_export, pattern_kwargs_for_export)
    # Build pattern replacement
    replace_pattern_gm = get_pattern_node(
        pattern_name, pattern, pattern_args_for_export, pattern_kwargs_for_export
    )
    print("check replacement gm")
    replace_pattern_gm.graph.print_tabular()
    print("check pattern gm")
    pattern_ep.graph_module.graph.print_tabular()
    # Eliminate placeholder for const, which is dangling, and trgerring assertion in matching
    eliminate_dangling_arg(pattern_ep.graph_module.graph)
    matches = subgraph_rewriter.replace_pattern_with_filters(
        exported_ep.graph_module,
        pattern_ep.graph_module,
        replace_pattern_gm,
        ignore_literals=True,
    )
    print("check matches")
    print(matches)
    capture_const = (not isinstance(pattern_args, Tuple)) or (constant_fx_node_name is not None)
    if capture_const:
        constant_node_mapping = constant_fx_node_name
        if not isinstance(pattern_args, Tuple):
            constant_node_mapping = find_constant_arg_mapping(pattern, pattern_args, pattern_kwargs)
        assert constant_node_mapping is not None
        print(constant_node_mapping)
        extract_constant_from_matched_pattern(matches, constant_node_mapping)
    exported_ep.graph_module.graph.print_tabular()
    return exported_ep
