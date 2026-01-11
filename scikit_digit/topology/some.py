from topology_io import build_fully_connected_input_output, save_topology_npz

topo = build_fully_connected_input_output(Nin=64, K=10, copies_per_pair=3)
save_topology_npz("scikit_digit/topology/digits_8x8_dense_io_x3.npz", topo)