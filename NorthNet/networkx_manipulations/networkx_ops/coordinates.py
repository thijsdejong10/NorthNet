def add_coords_to_network(network, pos):
    xmin = np.mean([pos[p][0] for p in pos])
    ymin = np.mean([pos[p][1] for p in pos])
    for n in network.nodes:
        if n in pos:
            network.nodes[n]['pos'] = pos[n]
        else:
            network.nodes[n]['pos'] = (xmin, ymin)

    return network

def set_network_coordinates(G, coords_file):
    '''
    Adds coordinates to node attributes from a .csv file.

    Parameters
    ----------
    G: networkx DiGraph
        Network
    coords_file: str
        Path to file containing node coordinates (format: compound, x, y newline)

    '''
    # Build coordinates list
    spec_coords = {}
    with open(coords_file, "r") as f:
        for line in f:
            ln = line.strip("\n")
            ln = ln.split(",")
            if ln[0].strip('"') in spec_coords:
                pass
            else:
                spec_coords[ln[0].strip('"')] = tuple([float(x) for x in ln[1:]])

    for n in G.nodes:
        if n in spec_coords:
            G.nodes[n]["pos"] = spec_coords[n]
        else:
            G.nodes[n]["pos"] = (randint(0,100),randint(0,100))

    return G

def get_network_coordinates(G):

    net_lines = []
    for e in G.edges:
        for n in e:
            net_lines.append(G.nodes[n]["pos"])
        net_lines.append((np.nan,np.nan))
    net_lines = np.array(net_lines)
    net_lines = net_lines.T

    return net_lines

def normalise_network_coordinates(G):

    coords = network_view.get_network_coordinates(G)

    net_width = (np.nanmax(coords[0])-np.nanmin(coords[0]))
    net_height = (np.nanmax(coords[1])-np.nanmin(coords[1]))

    for n in G.nodes:
        pos = G.nodes[n]['pos']
        a = pos[0]/net_width
        b = pos[1]/net_height
        G.nodes[n]['pos'] = (a,b)
