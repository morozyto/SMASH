from optparse import OptionParser


def parse_options():
    parser = OptionParser()

    # manage computation params
    parser.add_option("-c", "--parallel-count", type="int", default=4, dest="parallel_count")

    # points generation options
    parser.add_option("-n", action="store", type="int", default=250, dest="edge_size")
    parser.add_option("--max-random-num", type="int", default=100, dest="max_random_num")
    parser.add_option("--min-random-num", type="int", default=0, dest="min_random_num")

    # tree building options
    parser.add_option("-f", "--farfield-tolerance", action="store", type="float", default=10 ** -7, dest="farfield_tolerance")
    parser.add_option("-s", "--svd-tolerance", action="store", type="float", default=10 ** -7, dest="svd_tolerance")
    parser.add_option("-m", "--max-node-values", type="int", default=30, dest="max_values_in_node")

    # logs options
    parser.add_option("--write-logs-stdout", action='store_true', dest="stdout_logs")
    parser.add_option("-l", "--log-file", type="string", default='log', dest="log_file")
    parser.add_option("-d", "--log-dir", type="string", default='logs', dest="log_dir")
    parser.add_option("-p", "--np-precision", action="store", type="int", default=3, dest="precision")
    parser.add_option("-v", action='store_true', dest="debug_level")

    (options, _) = parser.parse_args()
    return options
