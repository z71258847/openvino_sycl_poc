
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
from operator import add
from functools import reduce
from math import ceil
from statistics import mean
from tabulate import tabulate

def get_output_info(tree, layer_id):
    for layer in tree.getroot()[0]:
        if layer_id == int(layer.get('id')):
            return layer[0].get('outputLayouts'), layer[0].get('outputPrecisions')
    return None, None

def find_real_layer(tree, not_executed, layer_id):
    if int(layer_id) in not_executed:
        for edge in tree.getroot()[1]:
            input_id = edge.get('from-layer')
            output_id = edge.get('to-layer')
            if output_id == layer_id:
                if input_id in not_executed:
                    return find_real_layer(tree, not_executed, int(input_id))
                else:
                    return int(input_id)
    else:
        return int(layer_id)

def shorten_layout(layout):
    if layout is not None:
        return layout.replace('fsv', 'f').replace('bsv','b')
    else:
        return None

def parse_xml(args):
    tree = ET.parse(args.xml)
    # if tree.getroot().get('name') != "runtime_gpu_graph":
        # print("[ERROR] input graph is not runtime gpu graph")
        # return

    not_executed = set()
    ignored_field = 'not_executed'
    if args.show_optimized_layer:
        ignored_field = 'no_field_left_behind'
    else:
        for layer in tree.getroot()[0]:
            mcs = layer[0].get('execTimeMcs')
            if mcs == ignored_field:
                not_executed.add(int(layer.get('id')))

    list_depth = []
    total_eff = 0
    num_layers = 0
    total_mcs = 0
    input_map = {'0': [None]}
    per_type = {}

    for edge in tree.getroot()[1]:
        input_id = edge.get('from-layer')
        output_id = edge.get('to-layer')

        if output_id not in input_map.keys():
            input_map[output_id] = []

        input_map[output_id].append(find_real_layer(tree, not_executed, input_id))

    max_name_len = max([len(layer.get('name')) if layer[0].get('execTimeMcs') != ignored_field else 0 for layer in tree.getroot()[0]]) + 1
    max_name_len = min(max_name_len, args.max_name_len)

    table = []

    for layer in tree.getroot()[0]:
        name = layer.get('name')
        data = layer[0]
        layer_type = layer.get('type')

        mcs = data.get('execTimeMcs')
        if mcs == ignored_field:
            continue

        if layer_type not in per_type:
            per_type[layer_type] = [0, 0, 0]

        try:
            if ':' in data.get('primitiveType'):
                per_type[layer_type][0] += int(mcs)
            else:
                per_type[layer_type][1] += int(mcs)
            per_type[layer_type][2] += int(mcs)
        except:
            # For not_executed primitive, we cannot calculate time.
            pass

        output = layer.find('output')
        if output is None:
            output_dim = "N/A"
        else:
            output_dim = [int(output[0][i].text) for i in range(len(output[0]))]

        input0 = layer.find('input')
        if input0 is None:
            input0_dim = "N/A"
        else:
            input0_dim = [int(input0[0][i].text) for i in range(len(input0[0]))]

        if layer.get('id') not in input_map:
            input_map[layer.get('id')] = [None]
        input0_layout, input0_precision = get_output_info(tree, input_map[layer.get('id')][0])

        entry = [layer.get('id') if not args.no_number else '']
        if not args.no_name:
            entry.append(name if len(name) <= max_name_len else name[:max_name_len-3] + '...')
        entry.append(layer.get('type'))
        entry.append(mcs if not args.no_number else '')

        if args.show_input_info:
            entry += [str(input_map[layer.get('id')]),
                      str(input0_dim),
                      str(shorten_layout(input0_layout)),
                      str(input0_precision)]

        entry += [
                    str(output_dim),
                    shorten_layout(data.get('outputLayouts')),
                    data.get('outputPrecisions'),
                    data.get('primitiveType')]

        try:
            total_mcs += int(mcs)
        except:
            pass

        table.append(entry)

    headers = ['id']
    if not args.no_name:
        headers.append('name')

    if args.show_input_info:
        headers += ['type', 'Mcs', 'inputs', 'input0 shape', 'in0 layout', 'in0 pr', 'out shape', 'out layout', 'out pr', 'primitive']
    else:
        headers += ['type', 'Mcs', 'out shape', 'out layout', 'out prec', 'primitive']

    if args.show_percentage:
        headers.append('Mcs%')
        for row in table:
            layer_mcs = int(row[3])
            percentage = layer_mcs / total_mcs if layer_mcs > 0 else 0
            percentage = percentage * 100
            row.append(f'{percentage:.2f}')

    print(tabulate(table, headers=headers, tablefmt="github"))

    print(f'Total execution mcs: {total_mcs}')
    if args.disable_summary != True:
        print()
        print('Per-type summary')
        table = [ [k] + per_type[k] for k in per_type.keys()]
        total = ['Total'] + [ sum([e[1] for e in table]), sum([e[2] for e in table]), sum([e[3] for e in table]) ]
        table.append(total)

        print(tabulate(table, headers=['OneDNN', 'clDNN', 'Total']))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Tool to format exec graph dump from IE. You can dump exec graph from benchmark_app with -exec_graph_path option.')
    parser.add_argument(dest='xml', help='XML of exec graph')
    parser.add_argument('-m', '--max_name_len', help='Maximum length of a layer name', default=40, type=int)
    parser.add_argument('-a', '--show_optimized_layer', help='Show all layers', action='store_true')
    parser.add_argument('-i', '--show_input_info', help='Show information about input layers.', action='store_true')
    parser.add_argument('-d', '--disable_summary', help='Don\'t show summary', action='store_true')
    parser.add_argument('-p', '--show_percentage', help='Show percentage of total time', action='store_true')
    parser.add_argument('-n', '--no_number', help='Do not print layer id and execution time', action='store_true')
    parser.add_argument('--no_name', help='Do not print layer name', action='store_true')

    args = parser.parse_args()

    parse_xml(args)


if __name__ == '__main__':
    main()
