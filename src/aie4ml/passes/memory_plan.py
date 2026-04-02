# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..aie_types import AIEDataType
from ..ir import OpNode, get_backend_context, input_role
from .base import AIEPass
from .boundary_sharding import (
    graph_input_port_descriptor,
    graph_input_writer_port_descriptor,
    localize_graph_io_descriptor,
)
from .utils import sanitize_identifier


@dataclass
class _Connection:
    producer: Optional[OpNode]
    consumer: Optional[OpNode]
    producer_group: str
    consumer_group: str
    tensor: str
    external_kind: Optional[str] = None


@dataclass
class _EdgeEntry:
    tensor: str
    producer: Optional[OpNode]
    producer_group: str
    producer_ports: int
    consumers: List[_Connection] = field(default_factory=list)
    graph_output: bool = False
    producer_port_ids: List[int] = field(default_factory=list)
    producer_tensor_port_base: int = 0
    consumer_port_ids: List[int] = field(default_factory=list)
    shard_dim: Optional[int] = None
    shard_port_stride: Optional[int] = None
    shard_dim_base: int = 0
    shard_dim_size: Optional[int] = None
    shard_offset_base: List[int] = field(default_factory=list)
    shard_buffer_dimension: List[int] = field(default_factory=list)
    graph_input_port_descs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    graph_input_writer_port_descs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    shard_index: int = 0
    shard_count: int = 1


class BuildMemoryPlan(AIEPass):
    def __init__(self):
        self.name = 'plan_memory'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        ctx.ir.physical.plan = _CodegenPlanner(ctx).build(list(ctx.ir.logical))
        return True


class CollectMemoryEntries(AIEPass):
    def __init__(self):
        self.name = 'collect_memory_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        planner = _CodegenPlanner(ctx)
        state = planner.collect(list(ctx.ir.logical))
        ctx.ir.physical.plan = {'_memory_plan_state': state}
        return True


class MaterializeMemoryPlan(AIEPass):
    def __init__(self):
        self.name = 'materialize_memory_plan'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        planner = _CodegenPlanner(ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        ctx.ir.physical.plan = planner.materialize(state)
        return True


class _CodegenPlanner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.device = ctx.device

        self.buffers = []
        self.direct_edges = []
        self.layer_indices = {}
        self._next_graph_output_port = 0
        self._max_graph_input_port = -1
        self._buffer_seq: Dict[str, int] = {}

    def build(self, nodes):
        state = self.collect(nodes)
        return self.materialize(state)

    def collect(self, nodes):
        idx = 0
        for n in nodes:
            if self._kernel_inst(n):
                idx += 1
                self.layer_indices[n.name] = idx

        conns = self._collect_connections(nodes)
        entries = self._group_edges(conns)
        return {
            'layer_indices': dict(self.layer_indices),
            'entries': entries,
        }

    def materialize(self, state):
        self.buffers = []
        self.direct_edges = []
        self.layer_indices = dict(state['layer_indices'])
        self._next_graph_output_port = 0
        self._max_graph_input_port = -1
        self._buffer_seq = {}

        for entry in state['entries']:
            self._materialize_entry(entry)

        return {
            'buffers': self.buffers,
            'direct_edges': self.direct_edges,
            'graph_input_count': self._max_graph_input_port + 1,
            'graph_output_count': self._next_graph_output_port,
        }

    # -------------------------------------------------------------------------
    # Connections
    # -------------------------------------------------------------------------

    def _collect_connections(self, nodes: List[OpNode]) -> List[_Connection]:
        producers: Dict[str, Tuple[OpNode, str]] = {}

        # collect producers
        for n in nodes:
            inst = self._kernel_inst(n)
            if not inst:
                continue
            for t in getattr(n, 'outputs', []):
                tname = t.name
                pg = inst.config.ports.outputs[tname].group
                producers[tname] = (n, pg)

        connections: List[_Connection] = []
        seen_outputs: set[str] = set()
        graph_output_names = set(self.ctx.ir.logical.output_tensor_names)

        # inputs — skip parameter tensors
        for n in nodes:
            inst = self._kernel_inst(n)
            if not inst:
                continue
            for t in getattr(n, 'inputs', []):
                if t.is_parameter:
                    continue
                tname = t.name
                cg = inst.config.ports.inputs[tname].group
                if tname in producers:
                    p, pg = producers[tname]
                    connections.append(_Connection(p, n, pg, cg, tname))
                    seen_outputs.add(tname)
                else:
                    connections.append(_Connection(None, n, 'graph_input', cg, tname, 'input'))

        # graph outputs
        for n in nodes:
            inst = self._kernel_inst(n)
            if not inst:
                continue
            for t in getattr(n, 'outputs', []):
                tname = t.name
                if tname not in graph_output_names and tname in seen_outputs:
                    continue
                pg = inst.config.ports.outputs[tname].group
                connections.append(_Connection(n, None, pg, 'graph_output', tname, 'output'))

        return connections

    # -------------------------------------------------------------------------
    # Group edges
    # -------------------------------------------------------------------------

    def _group_edges(self, connections: Iterable[_Connection]) -> List[_EdgeEntry]:
        grouped: Dict[Tuple[str, str], _EdgeEntry] = {}

        for c in connections:
            key = (c.tensor, c.producer_group)
            if key not in grouped:
                grouped[key] = _EdgeEntry(
                    tensor=c.tensor,
                    producer=c.producer,
                    producer_group=c.producer_group,
                    producer_ports=self._producer_port_count(c.producer, c.tensor),
                )

            e = grouped[key]

            if c.external_kind == 'output':
                e.graph_output = True
            else:
                e.consumers.append(c)
                if c.external_kind == 'input':
                    e.producer_ports = max(
                        e.producer_ports,
                        self._consumer_port_count(c.consumer, c.tensor),
                    )

        return list(grouped.values())

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize_entry(self, entry: _EdgeEntry):
        if len(entry.consumers) > 1:
            raise RuntimeError(f'{entry.tensor}: materializer requires at most one consumer per entry.')
        if entry.consumers and entry.graph_output:
            raise RuntimeError(f'{entry.tensor}: materializer does not accept mixed consumer + graph_output entry.')
        if not entry.producer_port_ids:
            raise RuntimeError(f'{entry.tensor}: missing producer_port_ids; run port-limit legalization first.')
        graph_input_generic = entry.producer is None and bool(entry.graph_input_port_descs)
        if not graph_input_generic and (
            entry.shard_dim is None or entry.shard_port_stride is None or entry.shard_dim_size is None
        ):
            raise RuntimeError(f'{entry.tensor}: missing shard metadata; run port-limit legalization first.')

        p_ports = [int(x) for x in entry.producer_port_ids]
        c_ports = [int(x) for x in entry.consumer_port_ids]

        route = self._route_policy(entry)
        direct_eligible = (
            entry.shard_count == 1
            and entry.producer
            and not entry.graph_output
            and len(entry.consumers) == 1
            and len(p_ports) == len(c_ports)
            and entry.consumers[0].consumer.traits['io_view'].data['inputs'][entry.tensor].get('perm') is None
            and self._direct_transport_supported(entry, p_ports, c_ports)
        )

        if route == 'direct':
            if not direct_eligible:
                raise RuntimeError(f'{entry.tensor}: io_route=direct requested but direct transport is not legal.')
            self._emit_direct(entry, p_ports, c_ports)
            return

        if direct_eligible and route != 'memtile':
            self._emit_direct(entry, p_ports, c_ports)
        else:
            self._emit_memtile(entry, p_ports, c_ports)

    @staticmethod
    def _normalized_direct_staging(desc: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if desc is None:
            return None
        data = {k: v for k, v in desc.items() if k not in ('access', 'boundary_dimension')}
        if 'io_boundary_dimension' in data and 'boundary_dimension' not in data:
            data['boundary_dimension'] = data['io_boundary_dimension']
        return data

    def _direct_transport_supported(self, entry: _EdgeEntry, p_ports: List[int], c_ports: List[int]) -> bool:
        if entry.producer is None or len(entry.consumers) != 1 or len(p_ports) != len(c_ports):
            return False
        src_inst = self._kernel_inst(entry.producer)
        dst_inst = self._kernel_inst(entry.consumers[0].consumer)
        for p_port, c_port in zip(p_ports, c_ports):
            src_desc = src_inst.variant.describe_output_staging(
                entry.producer, src_inst.config, entry.tensor, int(p_port), None
            )
            dst_desc = dst_inst.variant.describe_input_staging(
                entry.consumers[0].consumer,
                dst_inst.config,
                entry.tensor,
                int(c_port),
                None,
                entry.producer,
            )
            if self._normalized_direct_staging(src_desc) != self._normalized_direct_staging(dst_desc):
                return False
        return True

    def _route_policy(self, entry: _EdgeEntry) -> str:
        if entry.producer is None or entry.graph_output:
            return 'memtile'

        modes = set()
        p_inst = self._kernel_inst(entry.producer)
        p_mode = p_inst.config.io_route.get('outputs', {}).get(entry.tensor)
        if p_mode:
            modes.add(str(p_mode))

        for c in entry.consumers:
            c_inst = self._kernel_inst(c.consumer)
            c_mode = c_inst.config.io_route.get('inputs', {}).get(entry.tensor)
            if c_mode:
                modes.add(str(c_mode))

        bad = [m for m in modes if m not in ('direct', 'memtile', 'auto')]
        if bad:
            raise ValueError(f'{entry.tensor}: unsupported io_route mode(s) {bad}.')
        if 'memtile' in modes:
            return 'memtile'
        if modes == {'direct'}:
            return 'direct'
        return 'auto'

    # ------------------------------------------------------------------
    # Direct
    # ------------------------------------------------------------------

    def _emit_direct(self, entry, p_ports, c_ports):
        p = entry.producer
        c = entry.consumers[0]

        for p_port, c_port in zip(p_ports, c_ports):
            self.direct_edges.append(
                {
                    'source': f'{sanitize_identifier(p.name)}.{entry.producer_group}[{int(p_port)}]',
                    'target': f'{sanitize_identifier(c.consumer.name)}.{c.consumer_group}[{int(c_port)}]',
                    'tensor': entry.tensor,
                }
            )

    # ------------------------------------------------------------------
    # Memtile
    # ------------------------------------------------------------------

    def _emit_memtile(self, entry, p_ports, c_ports):
        if entry.producer:
            inst = self._kernel_inst(entry.producer)
            base = inst.variant.describe_output_staging(entry.producer, inst.config, entry.tensor, 0, None)
            shard_dim = int(entry.shard_dim)
            port_stride = int(entry.shard_port_stride)
            unit_base_dim0 = int(entry.shard_dim_base)
            full_dims = list(base['buffer_dimension'])
            buf_dims = list(full_dims)
            buf_dims[shard_dim] = int(entry.shard_dim_size)
        else:
            if entry.graph_input_port_descs:
                base = graph_input_port_descriptor(entry, p_ports[0])
                buf_dims = list(entry.shard_buffer_dimension)
            else:
                base = self._graph_input_writer_descriptor(entry)
                shard_dim = int(entry.shard_dim)
                port_stride = int(entry.shard_port_stride)
                unit_base_dim0 = int(entry.shard_dim_base)
                full_dims = list(base['buffer_dimension'])
                buf_dims = list(full_dims)
                buf_dims[shard_dim] = int(entry.shard_dim_size)

        name = self._next_buffer_name(entry)
        buffer = {
            'name': name,
            'dimension': buf_dims,
            'num_buffers': 2,
            'ctype': self._buffer_ctype(entry),
            'writers': [],
            'readers': [],
            'tensor': entry.tensor,
        }

        base_p = p_ports[0]
        for slot, p in enumerate(p_ports):
            if entry.producer is None:
                if entry.graph_input_port_descs:
                    desc = localize_graph_io_descriptor(
                        graph_input_writer_port_descriptor(entry, int(p)),
                        list(entry.shard_offset_base),
                        list(buf_dims),
                    )
                else:
                    desc = self._graph_input_writer_descriptor(entry)
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] = (int(p) - int(base_p)) * int(port_stride)
                self._max_graph_input_port = max(self._max_graph_input_port, int(p))
            else:
                inst = self._kernel_inst(entry.producer)
                desc = inst.variant.describe_output_staging(entry.producer, inst.config, entry.tensor, p, buf_dims)
                desc['buffer_dimension'] = list(buf_dims)
                desc['offset'][shard_dim] -= int(unit_base_dim0)

            buffer['writers'].append(
                {
                    'source': self._producer_endpoint(entry.producer, entry.producer_group, p),
                    'source_type': self._producer_endpoint_meta(entry.producer, entry.producer_group, p)[0],
                    'source_endpoint': self._producer_endpoint_meta(entry.producer, entry.producer_group, p)[1],
                    'target': f'{name}.in[{slot}]',
                    'descriptor': desc,
                    'staging': self._graph_input_staging(entry, int(p)) if entry.producer is None else None,
                    'dtype': self._graph_input_dtype(entry).to_dict() if entry.producer is None else None,
                }
            )

        if entry.consumers:
            consumer_conn = entry.consumers[0]
            for local_out, i in enumerate(c_ports):
                if entry.producer is None and entry.graph_input_port_descs:
                    desc = localize_graph_io_descriptor(
                        graph_input_port_descriptor(entry, int(base_p + local_out)),
                        list(entry.shard_offset_base),
                        list(buf_dims),
                    )
                else:
                    inst = self._kernel_inst(consumer_conn.consumer)
                    desc = inst.variant.describe_input_staging(
                        consumer_conn.consumer, inst.config, entry.tensor, i, buf_dims, entry.producer
                    )
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] -= int(unit_base_dim0)
                    if int(entry.shard_count) > 1:
                        desc['boundary_dimension'] = list(buf_dims)

                buffer['readers'].append(
                    {
                        'source': f'{name}.out[{local_out}]',
                        'target': (
                            f'{sanitize_identifier(consumer_conn.consumer.name)}.'
                            f'{consumer_conn.consumer_group}[{i}]'
                        ),
                        'target_type': 'op_impl',
                        'target_endpoint': {
                            'op_impl': consumer_conn.consumer.name,
                            'op_impl_id': sanitize_identifier(consumer_conn.consumer.name),
                            'group': consumer_conn.consumer_group,
                            'port': int(i),
                        },
                        'descriptor': desc,
                    }
                )

        if entry.graph_output:
            reader_base = len(buffer['readers'])
            for slot, local_port in enumerate(p_ports):
                graph_port = self._next_graph_output_port
                self._next_graph_output_port += 1
                desc = self._graph_output_reader_descriptor(entry, local_port, buf_dims, unit_base_dim0=unit_base_dim0)
                buffer['readers'].append(
                    {
                        'source': f'{name}.out[{reader_base + slot}]',
                        'target': f'ofm[{graph_port}]',
                        'target_type': 'plio',
                        'target_endpoint': {'name': 'ofm', 'port': int(graph_port), 'op_impl_port': int(local_port)},
                        'descriptor': desc,
                        'staging': self._graph_output_staging(entry, int(local_port)),
                        'dtype': self._graph_output_dtype(entry).to_dict(),
                    }
                )

        self.buffers.append(buffer)

    # -------------------------------------------------------------------------
    # Graph IO descriptors
    # -------------------------------------------------------------------------

    def _graph_input_writer_descriptor(self, entry: _EdgeEntry) -> Dict[str, Any]:
        c = entry.consumers[0].consumer
        inst = self._kernel_inst(c)
        base = inst.variant.describe_input_staging(c, inst.config, entry.tensor, 0, None, None)

        io_tile = list(base['io_tiling_dimension'])

        return {
            'access': 'write',
            'buffer_dimension': list(base['buffer_dimension']),
            'tiling_dimension': io_tile,
            'io_tiling_dimension': list(io_tile),
            'io_boundary_dimension': list(base['io_boundary_dimension']),
            'offset': [0 for _ in io_tile],
            'slice_dimension': int(base['slice_dimension']),
            'feature_dimension': int(base['feature_dimension']),
            'independent_dimension': int(base['independent_dimension']),
        }

    def _graph_output_reader_descriptor(
        self,
        entry: _EdgeEntry,
        port: int,
        buf_dims: List[int],
        unit_base_dim0: int,
    ) -> Dict[str, Any]:
        p = entry.producer
        inst = self._kernel_inst(p)
        base = inst.variant.describe_output_staging(p, inst.config, entry.tensor, port, buf_dims)
        shard_dim = int(base['slice_dimension'])
        io_tile = list(base['io_tiling_dimension'])
        io_boundary = list(base['io_boundary_dimension'])

        offset = list(base['offset'])
        offset[shard_dim] -= int(unit_base_dim0)
        boundary = list(io_boundary)
        boundary[shard_dim] = min(int(buf_dims[shard_dim]), max(0, int(io_boundary[shard_dim]) - int(unit_base_dim0)))
        return {
            'access': 'read',
            'buffer_dimension': list(buf_dims),
            'tiling_dimension': io_tile,
            'io_tiling_dimension': list(io_tile),
            'io_boundary_dimension': list(io_boundary),
            'offset': offset,
            'boundary_dimension': boundary,
            'slice_dimension': int(base['slice_dimension']),
            'feature_dimension': int(base['feature_dimension']),
            'independent_dimension': int(base['independent_dimension']),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _kernel_inst(self, node):
        return self.ctx.ir.execution.get(node.name) if node else None

    def _producer_port_count(self, node, tensor):
        if node is None:
            return 1
        return self._kernel_inst(node).config.ports.outputs[tensor].count

    def _consumer_port_count(self, node, tensor):
        return self._kernel_inst(node).config.ports.inputs[tensor].count

    def _producer_endpoint(self, node, group, port):
        return f'ifm[{port}]' if node is None else f'{sanitize_identifier(node.name)}.{group}[{port}]'

    def _producer_endpoint_meta(self, node, group, port):
        if node is None:
            return 'plio', {'name': 'ifm', 'port': int(port)}
        return 'op_impl', {
            'op_impl': node.name,
            'op_impl_id': sanitize_identifier(node.name),
            'group': group,
            'port': int(port),
        }

    def _consumer_endpoint_meta(self, node, group, port):
        if node is None:
            return 'plio', {'name': 'ofm', 'port': int(port)}
        return 'op_impl', {
            'op_impl': node.name,
            'op_impl_id': sanitize_identifier(node.name),
            'group': group,
            'port': int(port),
        }

    def _graph_input_role(self, entry: _EdgeEntry) -> str:
        consumer = entry.consumers[0].consumer
        role = input_role(consumer, entry.tensor)
        return role or 'lhs'

    def _buffer_ctype(self, entry):
        if entry.producer is None:
            c = entry.consumers[0].consumer
            role = self._graph_input_role(entry)
            suffix = 'rhs_t' if role == 'rhs' else 'data_t'
            return f'typename Cfg{self.layer_indices[c.name]}::{suffix}'
        return f'typename Cfg{self.layer_indices[entry.producer.name]}::result_t'

    def _graph_input_dtype(self, entry: _EdgeEntry) -> AIEDataType:
        consumer = entry.consumers[0].consumer
        inst = self._kernel_inst(consumer)
        role = self._graph_input_role(entry)
        return inst.config.parameters.precision[role]

    def _graph_input_staging(self, entry: _EdgeEntry, port: int) -> Dict[str, Any]:
        if entry.graph_input_port_descs:
            return graph_input_writer_port_descriptor(entry, int(port))
        desc = self._graph_input_writer_descriptor(entry)
        shard_dim = int(entry.shard_dim)
        port_stride = int(entry.shard_port_stride)
        tensor_local_port = int(port) - int(entry.producer_tensor_port_base)
        desc['offset'][shard_dim] = int(tensor_local_port) * int(port_stride)
        return desc

    def _graph_output_dtype(self, entry: _EdgeEntry) -> AIEDataType:
        producer = entry.producer
        inst = self._kernel_inst(producer)
        return inst.config.parameters.precision['output']

    def _graph_output_staging(self, entry: _EdgeEntry, port: int) -> Dict[str, Any]:
        producer = entry.producer
        inst = self._kernel_inst(producer)
        return inst.variant.describe_output_staging(producer, inst.config, entry.tensor, port, None)

    def _next_buffer_name(self, entry: _EdgeEntry):
        base = sanitize_identifier(entry.tensor)
        suffix = ''
        if int(entry.shard_count) > 1:
            suffix += f'_u{int(entry.shard_index)}'
        fan_index = getattr(entry, 'fan_index', None)
        if fan_index is not None:
            suffix += f'_fan{int(fan_index)}'

        key = f'{base}{suffix}'
        idx = self._buffer_seq.get(key, 0) + 1
        self._buffer_seq[key] = idx

        stem = f'buffer_{base}{suffix}'
        return stem if idx == 1 else f'{stem}_{idx}'
