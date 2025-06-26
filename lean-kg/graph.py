
import pathlib, os, re, logging, multiprocessing, networkx as nx, csv
from collections import deque
from typing import Union, List, Tuple

# Global containers for declaration names and the files they come from
all_decl_names = set() # all known full declaration names
decl_file = {} # maps full_name to source file
name_index = {} # short-name → full-name(s)

logging.basicConfig(level=logging.INFO, format="%(message)s")

def remove_comments(lean_code: str) -> str:
    """
    Removes single-line and block comments from a Lean code string.

    Args:
        lean_code: The input string containing Lean code.

    Returns:
        A string with comments removed.
    """
    code_no_line_comments = re.sub(r'--.*', '', lean_code) # Remove single-line comments (-- ...)
    result_chars = []
    i, depth = 0, 0 # Remove nested block comments (/- ... -/)
    while i < len(code_no_line_comments):
        if i+1 < len(code_no_line_comments) and code_no_line_comments[i] == '/' and code_no_line_comments[i+1] == '-':
            depth += 1
            i += 2
            continue
        if i+1 < len(code_no_line_comments) and code_no_line_comments[i] == '-' and code_no_line_comments[i+1] == '/' and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            result_chars.append(code_no_line_comments[i])
        i += 1
    return ''.join(result_chars)

def gather_decls(file_path:str,base_module:str):
    """
    Parse a Lean file and collect full names of all declarations and subdecls.

    Args:
        file_path: The path to the Lean file.
        base_module: The base module name for the file.

    Returns:
        A list of tuples, where each tuple is (full_name, kind) for a declaration.
    """
    decls = [] # list of tuples of the form (full_name, kind)
    try:
        text = open(file_path,'r',encoding='utf-8').read() # read raw file
    except Exception as e:
        logging.error(f"Could not read file {file_path}: {e}") # error handling for edge cases
        return decls

    code = remove_comments(text) # use previous function to strip comments
    lines = code.splitlines() # split the cleaned code into lines for scanning

    ns_stack = base_module.split('.') if base_module else [] # start namespace from module path
    current_ns = '.'.join(ns_stack) # active namespace for prefixing names
    i = 0  # line index

    while i < len(lines):
        line = lines[i].strip() # remove leading/trailing whitespace

        if line == "" or line.startswith( # skip blank lines
            ('import ','section ','namespace ','end ','open ',
             'open_locale ','open scoped ','notation ','macro ',
             'example ','attribute ')):

            if line.startswith('namespace '): # namespace push
                ns_name = line[len('namespace '):].strip()
                if ns_name:
                    ns_stack = ns_name.split('.') if '.' in ns_name else ns_stack + [ns_name]
                    current_ns = '.'.join(ns_stack)

            if line.startswith('end '):       # namespace pop
                end_name = line[len('end '):].strip()
                if end_name == '' and ns_stack:
                    ns_stack.pop()
                elif ns_stack and ns_stack[-1] == end_name:
                    ns_stack.pop()
                elif end_name in ns_stack:
                    while ns_stack and ns_stack[-1] != end_name:
                        ns_stack.pop()
                    if ns_stack and ns_stack[-1] == end_name:
                        ns_stack.pop()
                current_ns = '.'.join(ns_stack)

            if line == 'end': # shorthand: plain 'end'
                if ns_stack:
                    ns_stack.pop()
                    current_ns = '.'.join(ns_stack)
            i += 1
            continue

        if line.startswith('@['): # attribute lines (skip or merge)
            attr_end = line.find(']')
            if attr_end != -1 and attr_end < len(line)-1:
                line = line[attr_end+1:].lstrip()  # keep remainder if declaration is inline
            else:
                i += 1 # else skip to next line
                continue

        tokens = line.split() # tokenize line by words
        while tokens and tokens[0] in { # skip modifiers like 'noncomputable'
            "noncomputable","protected","private",
            "unsafe","partial","scoped"}:
            tokens.pop(0)
        if not tokens:
            i += 1
            continue

        keyword = tokens[0] # get main declaration type

        if keyword in {"def","lemma","theorem","abbrev","axiom","constant"}:
            if len(tokens) < 2:
                i += 1
                continue
            name = tokens[1].rstrip(':(') # extract decl name
            full_name = f"{current_ns}.{name}" if current_ns else name
            decls.append((full_name, keyword))
            all_decl_names.add(full_name)
            decl_file[full_name] = file_path
            i += 1

        elif keyword in {"structure","class"}:
            if len(tokens) < 2:
                i += 1
                continue
            name = tokens[1].rstrip(':(')
            full_name = f"{current_ns}.{name}" if current_ns else name
            kind = "class" if keyword == "class" else "structure"
            decls.append((full_name, kind))
            all_decl_names.add(full_name)
            decl_file[full_name] = file_path
            i += 1

            while i < len(lines): # parse structure fields
                nxt_line = lines[i].strip()
                if nxt_line == "" or nxt_line.startswith(('--','/-','deriving')):
                    i += 1
                    continue
                if nxt_line.startswith('@['): # skip field attributes
                    end_idx = nxt_line.find(']')
                    if end_idx != -1 and end_idx < len(nxt_line)-1:
                        nxt_line = nxt_line[end_idx+1:].lstrip()
                    else:
                        i += 1
                        continue
                if nxt_line.startswith(('def ','lemma ','theorem ',
                                        'inductive ','class ','structure ',
                                        'instance ','end','namespace ')):
                    break
                if ':' in nxt_line:           # field line
                    field_list = nxt_line.split(':',1)[0]
                    field_names = [fn.strip() for fn in field_list.split(',')]
                    for fn in field_names:
                        if fn:
                            field_full = f"{full_name}.{fn}"
                            decls.append((field_full, "field"))
                            all_decl_names.add(field_full)
                            decl_file[field_full] = file_path
                i += 1

        elif keyword == "instance": # instance (optional name)
            name = None
            if len(tokens) > 1:
                second = tokens[1]
                if second != ":" and not second.startswith(('(','{','extends','where')):
                    name = second.rstrip(':')
            if name:
                full_name = f"{current_ns}.{name}" if current_ns else name
                decls.append((full_name, "instance"))
                all_decl_names.add(full_name)
                decl_file[full_name] = file_path
            i += 1

        elif keyword == "inductive": # inductive type
            if len(tokens) < 2:
                i += 1
                continue
            name = tokens[1].rstrip(':')
            full_name = f"{current_ns}.{name}" if current_ns else name
            decls.append((full_name, "inductive"))
            all_decl_names.add(full_name)
            decl_file[full_name] = file_path
            i += 1

            while i < len(lines): # parse inductive constructors
                nxt_line = lines[i].strip()
                if nxt_line == "" or nxt_line.startswith(('--','/-','deriving')):
                    i += 1
                    continue
                if nxt_line.startswith('@['):
                    end_idx = nxt_line.find(']')
                    if end_idx != -1 and end_idx < len(nxt_line)-1:
                        nxt_line = nxt_line[end_idx+1:].lstrip()
                    else:
                        i += 1
                        continue
                if nxt_line.startswith('|'):
                    cons_def = nxt_line[1:].strip()
                    if cons_def == "":
                        i += 1
                        continue
                    cons_name = cons_def.split()[0].rstrip(':')
                    cons_full = f"{full_name}.{cons_name}"
                    decls.append((cons_full, "constructor"))
                    all_decl_names.add(cons_full)
                    decl_file[cons_full] = file_path
                    i += 1
                    continue
                break

        else:
            i += 1
    return decls

def resolve_reference(token:str,current_ns:str,open_prefixes:list):
    """
    Resolves a Lean token to a full declaration name.

    Args:
        token: The token to resolve.
        current_ns: The current namespace.
        open_prefixes: A list of opened namespaces.

    Returns:
        The full declaration name if resolved, otherwise None.
    """
    if token in all_decl_names: # if the token is already a full name
        return token
    if '.' in token: # if there is a dot in the token's name
        return token if token in all_decl_names else None
    candidates=name_index.get(token,[]) # find all decls sharing this short name
    if not candidates: # if there is no match
        return None
    if len(candidates)==1: # if there is only one match
        return candidates[0]
    for cand in candidates: # if there is one with the same namespace as current file
        if current_ns and cand.startswith(current_ns+"."):
            return cand
    for cand in candidates: # if there is one with any namespace opened via `open`
        for prefix in open_prefixes:
            if cand.startswith(prefix+"."):
                return cand
    return None

def parse_file_for_dependencies(args):
    """
    Parses a Lean file to extract declaration dependencies (edges).

    Args:
        args: A tuple containing (file_path, base_module).

    Returns:
        A list of tuples representing edges in the form (source, target).
    """
    file_path,base_module=args # unpack tuple
    try:text=open(file_path,'r',encoding='utf-8').read() # read file text
    except Exception as e: # error handling
        logging.error(f"Failed to open {file_path}: {e}")
        return []
    code=remove_comments(text) # strip comments
    lines=code.splitlines() # split into lines

    open_prefixes=[] # namespaces from `open`
    current_ns=base_module # namespace context
    edges=[]
    current_decl=None # declaration we are inside
    seen_deps=set() # avoid duplicate edges
    i=0 # line index

    while i<len(lines): # we iterate over the lines in the proof
        line=lines[i].strip() # trim whitespace

        if line=="" or line.startswith(('import ','section ','open_locale ',
                                        'open scoped ','notation ','macro ',
                                        'example ','attribute ')): # skip irrelevant lines
            if line.startswith('open '): # record opened modules
                for mod in line[len('open '):].split():
                    mod_prefix=mod.strip('()')
                    if mod_prefix and mod_prefix[0].isupper():
                        open_prefixes.append(mod_prefix)
            if line.startswith('namespace '): # push namespace
                ns=line[len('namespace '):].strip()
                current_ns=ns if '.' in ns else (current_ns+'.'+ns if current_ns else ns)
            if line.startswith('end '): # pop namespace
                end=line[len('end '):].strip()
                if end=="" or current_ns.endswith("."+end):
                    current_ns=current_ns[:-(len(end)+1)] if end else current_ns.rpartition('.')[0]
            if line=='end' and current_ns.rfind('.')!=-1:
                current_ns=current_ns.rpartition('.')[0]
            i+=1;continue

        if current_decl: # iterate over declatations
            if line.startswith(('def ','lemma ','theorem ','abbrev ','instance ',
                                'inductive ','structure ','class ','end ','namespace ')):
                current_decl=None;seen_deps.clear()
                continue
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9'_\.]*",line):
                if tok=='_' or tok.isnumeric():continue
                ref=resolve_reference(tok,current_ns,open_prefixes)
                if ref and ref!=current_decl and ref in all_decl_names and ref not in seen_deps:
                    edges.append((ref,current_decl)) # store edge (ref, decl)
                    seen_deps.add(ref)
            i+=1;continue

        if line.startswith('@['): # detect the start of a new declaration
            attr_end=line.find(']')
            if attr_end!=-1 and attr_end<len(line)-1:
                line=line[attr_end+1:].lstrip()
            else:i+=1;continue
        toks=line.split()
        while toks and toks[0] in {"noncomputable","protected","private",
                                   "unsafe","partial","scoped"}:
            toks.pop(0)
        if not toks:i+=1;continue
        kw=toks[0]
        if kw in {"def","lemma","theorem","abbrev","axiom","constant","instance",
                  "inductive","structure","class"}:
            name_tok=None
            if kw=="instance" and len(toks)>1 and toks[1] not in [":"] and                not toks[1].startswith(('(','{','extends','where')):
                name_tok=toks[1].rstrip(':')
            elif len(toks)>1:
                name_tok=toks[1].rstrip(':(')
            current_decl=f"{current_ns}.{name_tok}" if name_tok and current_ns else name_tok
            if kw in {"inductive","structure","class"}: # skip constructor/field lines
                i+=1
                while i<len(lines):
                    nxt=lines[i].strip()
                    if nxt=="" or nxt.startswith(('--','/-','deriving')):i+=1;continue
                    if nxt.startswith('@['):
                        end_idx = nxt.find(']')
                        if end_idx != -1 and end_idx < len(nxt)-1:
                            nxt = nxt[end_idx+1:].lstrip()
                        else:
                            i += 1
                            continue
                    if nxt.startswith('|') or (':' in nxt and not nxt.startswith(
                       ('def','lemma','theorem','inductive','structure','class',
                        'instance','end','namespace'))):
                        i+=1;continue
                    break
                continue
            else:
                i+=1;continue

        i+=1
    return edges

def build_dependency_graph(mathlib_dir:str,subdir_filter:str=None,num_workers:int=4):
    """
    Builds a directed multigraph representing the dependencies in mathlib4.

    Args:
        mathlib_dir: The root directory of mathlib4.
        subdir_filter: Optional subdirectory to filter by (e.g., "Algebra").
        num_workers: The number of worker processes to use for parsing.

    Returns:
        A NetworkX MultiDiGraph representing the dependency graph.
    """
    mathlib_dir=mathlib_dir.rstrip(os.sep) # normalise path
    base_len=len(mathlib_dir.rstrip(os.sep)+os.sep) # prefix length for rel-paths
    lean_files=[] # (file, module_name) pairs

    for root,_,files in os.walk(mathlib_dir): # walk directory tree
        for fname in files:
            if not fname.endswith(".lean"):continue # skip non-Lean files
            fpath=os.path.join(root,fname)
            if subdir_filter and subdir_filter not in os.path.relpath(fpath,mathlib_dir): # skip if not in requested subdir
                continue
            rel_path=fpath[base_len:].replace(os.sep,'.') # convert path → dotted module
            module_name=rel_path[:-5] # drop '.lean'
            lean_files.append((fpath,module_name))

    logging.info(f"Found {len(lean_files)} Lean files to parse.")
    for fpath,mod in lean_files: # parse each lean file and pull declarations (or stubs thereof)
        gather_decls(fpath,mod)

    global name_index # short-name → full-name(s)
    name_index={}
    for full in all_decl_names:
        short=full.split('.')[-1]
        name_index.setdefault(short,[]).append(full)

    with multiprocessing.Pool(processes=num_workers) as pool: # parallel edge extraction
        results=pool.map(parse_file_for_dependencies,lean_files) # run "parse_file_for_dependencies" on every Lean file in parallel
    all_edges=[e for lst in results for e in lst] # flatten list-of-lists

    G=nx.MultiDiGraph() # final graph
    for name in all_decl_names: # add nodes
        G.add_node(name,file=os.path.relpath(decl_file.get(name,""),mathlib_dir))
    for src,dst in all_edges: # add edges
        G.add_edge(src,dst)


    for decl in G.nodes: # iterate over every node
      G.nodes[decl]["name"] = decl # store display name
      if decl.endswith(".mk") or ".intro" in decl:
          kind = "Constructor"
      elif "axiom"    in decl.lower():
          kind = "Axiom"
      elif "theorem"  in decl.lower() or "lemma" in decl.lower():
          kind = "Theorem"
      elif "instance" in decl.lower():
          kind = "Instance"
      elif "def"      in decl.lower() or "abbrev" in decl.lower():
          kind = "Definition"
      else:
          kind = "Unknown"
      G.nodes[decl]["kind"] = kind
    return G

def expand_sample_graph(G, k=5, max_nodes=500):
    """
    Samples a subgraph by expanding from top-k highest-degree nodes.

    Args:
        G: The input NetworkX graph.
        k: The number of top-degree nodes to use as seeds.
        max_nodes: The maximum number of nodes in the sampled subgraph.

    Returns:
        A subgraph of G.
    """
    # 1. Pick top-k highest-degree nodes as seeds
    top_k_seeds = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:k]]

    visited = set(top_k_seeds)
    frontier = set()
    for seed in top_k_seeds:
        frontier |= set(G.predecessors(seed)) | set(G.successors(seed))

    # 2. Expand outward until max_nodes is reached
    while len(visited) < max_nodes and frontier:
        next_node = frontier.pop()
        visited.add(next_node)
        frontier |= set(G.predecessors(next_node)) | set(G.successors(next_node))
        frontier -= visited

    return G.subgraph(visited)

def compute_node_positions(G):
    """
    Computes the 2D positions of nodes using a spring layout, scaled by degree.

    Args:
        G: The input NetworkX graph.

    Returns:
        A dictionary mapping nodes to their (x, y) coordinates.
    """
    degree = dict(G.degree())  # Dictionary: node to degree (number of edges)
    strength = np.array([degree[n] for n in G.nodes()])  # Create array of degrees
    norm = (strength - strength.min()) / (np.ptp(strength) + 1e-9)  # Normalize degrees to [0,1] range
    pos = nx.spring_layout(G, seed=42, k=0.15, weight=None)  # Compute spring layout (ignoring edge weights)
    for n, (x, y) in pos.items():
        scale = 1 - 0.3 * norm[list(G.nodes()).index(n)]  # Scale high-degree nodes toward center
        pos[n] = (scale * x, scale * y)  # Update node position
    return pos  # Return dictionary: node → (x, y) coordinates

def color_nodes_by_type(G):
    """
    Generates node labels, hover names, and a color map based on declaration type.

    Args:
        G: The input NetworkX graph.

    Returns:
        A tuple containing:
        - labels: A list of node type labels.
        - hover_names: A list of node names for hover information.
        - color_discrete_map: A dictionary mapping types to colors.
    """
    color_discrete_map = {  # Color mapping
        "Theorem": "red", "Axiom": "blue",
        "Definition": "yellow", "Lemma": "green",
        "Instance": "orange", "Unknown": "gray"
    }
    labels = [G.nodes[n].get("kind", "Unknown") for n in G.nodes()]  # Get node kinds
    hover_names = [G.nodes[n].get("name", str(n)) for n in G.nodes()]  # Hover names
    return labels, hover_names, color_discrete_map

def color_nodes_by_edges(G):
    """
    Generates node degree list and hover names for coloring by edge count.

    Args:
        G: The input NetworkX graph.

    Returns:
        A tuple containing:
        - degree_list: A list of node degrees.
        - hover_names: A list of node names for hover information.
    """
    degrees = dict(G.degree())
    degree_list = [degrees[n] for n in G.nodes()]
    hover_names = [G.nodes[n].get("name", str(n)) for n in G.nodes()]
    return degree_list, hover_names

def propagated_degree_score(G, max_depth=10):
    """
    Computes a weighted degree score for each node based on neighbor degrees up to a max depth.

    Args:
        G: The input NetworkX graph.
        max_depth: The maximum depth to consider for neighbor degrees.

    Returns:
        A dictionary mapping nodes to their propagated degree scores.
    """
    scores = dict()
    degrees = dict(G.degree())
    for node in G.nodes():
        visited = set()
        queue = deque([(node, 0)])  # (current_node, depth)
        score = 0.0
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            weight = 1.0 / (2 ** depth)
            score += weight * degrees[current]
            if depth < max_depth:
                neighbors = G.neighbors(current)
                for neighbor in neighbors:
                    queue.append((neighbor, depth + 1))
        scores[node] = score
    return scores

def find_node_position(pos_dict, query):
    """
    Finds the position of a node in the graph based on a query string.

    Args:
        pos_dict: A dictionary mapping node names to their positions.
        query: The query string to search for (exact match, substring, or regex).

    Returns:
        A tuple containing the matched node name and its (x, y) position,
        or None if no match is found.
    """
    matches = []

    # 1. Exact match
    if query in pos_dict:
        matches = [(query, pos_dict[query])]

    # 2. Substring match
    if not matches:
        query_lower = query.lower()
        matches = [
            (name, pos_dict[name])
            for name in pos_dict
            if query_lower in name.lower()
        ]

    # 3. Regex fallback
    if not matches:
        try:
            pattern = re.compile(query, re.IGNORECASE)
            matches = [
                (name, pos_dict[name])
                for name in pos_dict
                if pattern.search(name)
            ]
        except re.error:
            print("Invalid regex pattern.")
            return None

    if not matches:
        print("No matches found.")
        return None

    name, (x, y) = matches[0]
    print(f"{name} → x = {x:.4f}, y = {y:.4f}")
    return matches[0]

def plot_by_type(G, x, y):
    """
    Creates a scatter plot of the graph nodes colored by declaration type.

    Args:
        G: The input NetworkX graph.
        x: A list of x-coordinates for the nodes.
        y: A list of y-coordinates for the nodes.

    Returns:
        A Plotly Figure object.
    """
    labels, hover_names, color_discrete_map = color_nodes_by_type(G)  # Get type labels
    fig = px.scatter(
        x=x, y=y,
        color=labels,
        hover_name=hover_names,
        color_discrete_map=color_discrete_map,
        labels={"color": "Kind"},
        title="Euclidean Graph Colored by Type"
    )
    return fig

def plot_by_weighted_edges(G, x, y):
    """
    Creates a scatter plot of the graph nodes colored by weighted edge scores.

    Args:
        G: The input NetworkX graph.
        x: A list of x-coordinates for the nodes.
        y: A list of y-coordinates for the nodes.

    Returns:
        A Plotly Figure object.
    """
    scores = propagated_degree_score(G)
    raw_scores = np.array([scores[n] for n in G.nodes()])

    transformed_scores = np.log1p(raw_scores) / np.log(10)
    normalized = (transformed_scores - transformed_scores.min()) / (np.ptp(transformed_scores) + 1e-9)

    vivid_gradient = [
        [0.0, "blue"],
        [0.3, "purple"],
        [0.6, "red"],
        [1.0, "yellow"]
    ]

    fig = px.scatter(
        x=x, y=y,
        color=normalized,
        color_continuous_scale=vivid_gradient,
        hover_name=[G.nodes[n].get("name", str(n)) for n in G.nodes()],
        hover_data={"Original Score": raw_scores},
        title="Graph of Mathlib4 Colored by Weighted Edge Score"
    )

    fig.update_coloraxes(colorbar_title=None)

    return fig

def plot_euclidean_graph(G, pos=None, color_by="type"):
    """
    Plots the graph in Euclidean 2-space using a specified layout and coloring.

    Args:
        G: The input NetworkX graph.
        pos: Optional dictionary of precomputed node positions.
        color_by: The criterion for coloring nodes ('type' or 'weighted_edges').

    Raises:
        ValueError: If color_by is not 'type' or 'weighted_edges'.
    """
    if isinstance(G, tuple) and pos is None:
      G, pos = G
    if pos is None: pos = compute_node_positions(G)  # Compute spring layout if missing

    x = [pos[n][0] for n in G.nodes()]  # x-coordinates
    y = [pos[n][1] for n in G.nodes()]  # y-coordinates

    if color_by == "type":
        fig = plot_by_type(G, x, y)
    elif color_by == "weighted_edges":
        fig = plot_by_weighted_edges(G, x, y)
    else:
        raise ValueError(f"color_by must be 'type' or 'weighted_edges'. Found: {color_by}")

    fig.update_traces(marker=dict(size=5, opacity=0.8))  # Node size, transparency
    fig.update_layout(
        height=600, width=600,
        yaxis_scaleanchor="x",  # Lock aspect ratio
        showlegend=False
    )
    fig.show()

def prune_graph(G,
                remove_isolated=True,
                exclude_defs=False,
                exclude_data=False,
                exclude_tactics=False,
                exclude_basic_files=False):
    keep_nodes = []

    for n, data in G.nodes(data=True):
        name = data.get("name", "").lower()
        file = data.get("file", "").lower()

        if exclude_defs and ("def" in name or "abbrev" in name):
            continue
        if exclude_data and "data" in file:
            continue
        if exclude_tactics and "tactic" in file:
            continue
        if exclude_basic_files and ".basic." in file:
            continue

        keep_nodes.append(n)

    pruned = G.subgraph(keep_nodes).copy()

    if remove_isolated:
        isolates = list(nx.isolates(pruned))
        pruned.remove_nodes_from(isolates)

    return pruned

def extract_subgraph_by_subdir(G_full, subdir_name):
    """
    Extracts a subgraph containing nodes related to a specific subdirectory.

    Args:
        G_full: The full NetworkX graph.
        subdir_name: The name of the subdirectory to filter by (e.g., "GroupTheory").

    Returns:
        A NetworkX MultiDiGraph representing the subgraph.
    """
    subdir_nodes = {
        n for n in G_full.nodes
        if subdir_name in G_full.nodes[n].get("file", "")
    }

    subdir_edges = [
        (u, v, d)
        for u, v, d in G_full.edges(data=True)
        if u in subdir_nodes or v in subdir_nodes
    ]

    G_subdir = nx.MultiDiGraph()
    G_subdir.add_edges_from(subdir_edges)

    for n in G_subdir.nodes:
        G_subdir.nodes[n].update(G_full.nodes[n])

    return G_subdir
