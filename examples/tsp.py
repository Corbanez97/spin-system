# type: ignore
import numpy as np
import tensorflow as tf
import networkx as nx
import osmnx as ox
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spin_engine.models.traveling_salesman import TravelingSalesmanSystem
from spin_engine.measurements import Energy
from spin_engine.dynamics import MetropolisHastings, Tracker

# --- CONFIGURATION ---
# PLACE = "Union Square, San Francisco, California"
PLACE = "Londrina, Paraná, Brasil"
L = 64
NUM_SPINS = L**2
REPLICAS = 32

# --- STEP 1: GEOSPATIAL DATA ---
print(f"Fetching map data for {PLACE}...")
G = ox.graph_from_address(PLACE, dist=5000, network_type="drive")
nodes_list = list(G.nodes)

# np.random.seed(42)
all_active_nodes = np.random.choice(nodes_list, L, replace=False).tolist()

print(f"Calculating distance matrix for {L} nodes...")
W = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        if i == j:
            W[i, j] = 0
        else:
            try:
                W[i, j] = nx.shortest_path_length(
                    G, all_active_nodes[i], all_active_nodes[j], weight='length') / 1000.0
            except nx.NetworkXNoPath:
                W[i, j] = 100.0

# --- STEP 2: SYSTEM INITIALIZATION & ANNEALING ---
# Higher A_val ensures constraints (valid TSP route) are prioritized over distance
max_dist = np.max(W)
A_val = max_dist * 10.0
B_val = 1.0

system = TravelingSalesmanSystem(
    cost_matrix=W,
    lattice_replicas=REPLICAS,
    constraint_strength=A_val,
    distance_strength=B_val
)

# Improved annealing schedule: More steps help convergence
# betas = [0.001, 0.01, 0.1, 1.0]
# Slower, smoother schedule
betas = list(np.logspace(np.log10(0.001), np.log10(5.0), num=20))
sweep_len = 2000
annealing_history = []

print("Starting simulated annealing...")
for b in betas:
    beta_val = tf.constant(b, dtype=tf.float32)
    simulation = MetropolisHastings(system)
    tracker = Tracker([Energy(system)])
    simulation.sweep(tracker, beta=beta_val, sweep_length=sweep_len,
                     num_disturbances=tf.constant(1))
    annealing_history.append(tracker.history['Energy'].numpy())
    print(f"Finished sweep for beta={b:.2f}")

full_energy_history = np.concatenate(annealing_history, axis=0)

# --- STEP 3: INTEGRATED DASHBOARD VISUALIZATION ---


def create_integrated_dashboard_v0(G, system, all_active_nodes, W, energy_history, betas_list, sweep_length):
    print("Generating Integrated Dashboard...")
    replicas_spins = system.spin_state.value().numpy()
    final_energies = energy_history[-1, :]
    total_steps = energy_history.shape[0]
    steps_x = np.arange(total_steps)
    mean_energy = np.mean(energy_history, axis=1)

    ref_node = all_active_nodes[0]
    center_lat, center_lon = G.nodes[ref_node]['y'], G.nodes[ref_node]['x']

    replica_data = []
    for r in range(len(replicas_spins)):
        spins = replicas_spins[r]
        x_mat = ((spins + 1) / 2).reshape(L, L)
        row_sums = np.sum(x_mat, axis=1)
        col_sums = np.sum(x_mat, axis=0)
        is_valid = np.allclose(row_sums, 1) and np.allclose(col_sums, 1)

        idx_seq = [np.argmax(x_mat[:, i]) for i in range(L)]
        try:
            ref_pos = idx_seq.index(0)
            idx_seq = np.roll(idx_seq, -ref_pos).tolist()
        except ValueError:
            continue

        route_nodes = [all_active_nodes[i] for i in idx_seq]
        route_nodes.append(route_nodes[0])
        dist = sum(W[all_active_nodes.index(u), all_active_nodes.index(v)]
                   for u, v in zip(route_nodes[:-1], route_nodes[1:]))

        replica_data.append({
            'id': r, 'energy': final_energies[r],
            'valid': "VALID" if is_valid else "INVALID",
            'route': route_nodes, 'distance': dist * 1000
        })

    replica_data.sort(key=lambda x: x['energy'])

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "map"}, {"type": "xy"}]],
        subplot_titles=("Optimal Hamiltonian Cycle", "Energy Evolution")
    )

    # --- 1. ENERGY TRACES (COLUMN 2) ---
    num_replicas_to_plot = min(50, energy_history.shape[1])
    for r in range(num_replicas_to_plot):
        fig.add_trace(go.Scatter(x=steps_x, y=energy_history[:, r], mode='lines',
                                 line=dict(
                                     color='rgba(150,150,150,0.15)', width=1),
                                 showlegend=False, hoverinfo='skip'), row=1, col=2)

    # ADDED: Placeholder for the Selected Replica Trace
    fig.add_trace(go.Scatter(x=steps_x, y=energy_history[:, replica_data[0]['id']],
                             mode='lines', line=dict(color='#AB63FA', width=3),
                             name='Selected Replica'), row=1, col=2)

    fig.add_trace(go.Scatter(x=steps_x, y=mean_energy, mode='lines',
                             line=dict(color='black', width=2, dash='dash'),
                             name='Mean Energy'), row=1, col=2)

    # --- 2. MAP TRACES (COLUMN 1) ---
    fig.add_trace(go.Scattermap(lat=[], lon=[], mode='lines', name="Route",
                                line=dict(width=4, color='#AB63FA')), row=1, col=1)
    fig.add_trace(go.Scattermap(lat=[G.nodes[n]['y'] for n in all_active_nodes],
                                lon=[G.nodes[n]['x']
                                     for n in all_active_nodes],
                                mode='markers', marker=dict(size=10, color='blue'),
                                name="Nodes"), row=1, col=1)

    # --- 3. ANIMATION FRAMES ---
    # NEW TRACE INDEX MAPPING:
    # 0 to (num_replicas_to_plot - 1) : Grey background lines
    # num_replicas_to_plot            : Selected Replica Energy (Highlighted)
    # num_replicas_to_plot + 1        : Mean Energy
    # num_replicas_to_plot + 2        : Map Route
    # num_replicas_to_plot + 3        : Map Nodes

    sel_energy_idx = num_replicas_to_plot
    route_idx = num_replicas_to_plot + 2
    node_idx = num_replicas_to_plot + 3

    frames = []
    slider_steps = []

    for rank, data in enumerate(replica_data[:100]):
        lats, lons = [], []
        for u, v in zip(data['route'][:-1], data['route'][1:]):
            try:
                path = nx.shortest_path(G, u, v, weight='length')
                lats.extend([G.nodes[n]['y'] for n in path] + [None])
                lons.extend([G.nodes[n]['x'] for n in path] + [None])
            except:
                lats.extend([G.nodes[u]['y'], G.nodes[v]['y'], None])
                lons.extend([G.nodes[u]['x'], G.nodes[v]['x'], None])

        frame_name = f"rank_{rank}"
        frames.append(go.Frame(
            data=[
                # Update Highlighted Energy Trace
                go.Scatter(x=steps_x, y=energy_history[:, data['id']]),
                # Update Map Route
                go.Scattermap(lat=lats, lon=lons,
                              name=f"Route ({data['valid']})"),
                # Keep Nodes visible
                go.Scattermap(lat=[G.nodes[n]['y'] for n in all_active_nodes],
                              lon=[G.nodes[n]['x'] for n in all_active_nodes])
            ],
            name=frame_name,
            traces=[sel_energy_idx, route_idx, node_idx]
        ))

        slider_steps.append({
            "args": [[frame_name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f"Rank {rank+1}", "method": "animate"
        })

    # --- UPDATED LAYOUT FOR DYNAMIC SCREEN FITTING ---
    fig.update_layout(
        autosize=True,      # Enables automatic resizing
        width=None,         # Removes fixed width
        height=None,        # Removes fixed height (uses container height)
        # Tighten margins for more screen space
        margin=dict(l=20, r=20, t=50, b=20),

        map_style="carto-positron",
        map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=13),
        sliders=[{
            "steps": slider_steps,
            "active": 0,
            "currentvalue": {"prefix": "Selection: "},
            "pad": {"t": 50}  # Add padding so slider doesn't overlap the map
        }],
        title_text=f"TSP Optimization Dashboard: {PLACE}",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_yaxes(title_text="System Energy (H)", row=1, col=2, type="log")
    fig.frames = frames

    return fig


def create_integrated_dashboard(G, system, all_active_nodes, W, energy_history):
    print("Generating Interactive Dashboard (Slider Mode)...")

    # --- CONFIG: COLORS ---
    C_VALID = '#AB63FA'   # Purple
    C_INVALID = '#EF553B'  # Red (Standard Plotly Red)

    # --- Data Preparation ---
    replicas_spins = system.spin_state.value().numpy()
    final_energies = energy_history[-1, :]
    total_steps = energy_history.shape[0]
    steps_x = np.arange(total_steps)
    mean_energy = np.mean(energy_history, axis=1)
    L = len(all_active_nodes)

    # Map Centroid & Static Node Coords
    ref_node = all_active_nodes[0]
    center_lat, center_lon = G.nodes[ref_node]['y'], G.nodes[ref_node]['x']
    node_lats = [G.nodes[n]['y'] for n in all_active_nodes]
    node_lons = [G.nodes[n]['x'] for n in all_active_nodes]

    # Process Replica Data
    replica_data = []
    for r in range(len(replicas_spins)):
        spins = replicas_spins[r]
        x_mat = ((spins + 1) / 2).reshape(L, L)
        row_sums = np.sum(x_mat, axis=1)
        col_sums = np.sum(x_mat, axis=0)
        is_valid = np.allclose(row_sums, 1) and np.allclose(col_sums, 1)

        idx_seq = [np.argmax(x_mat[:, i]) for i in range(L)]
        try:
            ref_pos = idx_seq.index(0)
            idx_seq = np.roll(idx_seq, -ref_pos).tolist()
        except ValueError:
            continue

        route_nodes = [all_active_nodes[i] for i in idx_seq]
        route_nodes.append(route_nodes[0])
        dist = sum(W[all_active_nodes.index(u), all_active_nodes.index(v)]
                   for u, v in zip(route_nodes[:-1], route_nodes[1:]))

        replica_data.append({
            'id': r, 'energy': final_energies[r],
            'valid': "VALID" if is_valid else "INVALID",
            'is_valid_bool': is_valid,
            'route': route_nodes, 'idx_seq': idx_seq,
            'distance': dist * 1000, 'x_mat': x_mat
        })

    replica_data.sort(key=lambda x: x['energy'])

    # --- SUBPLOT LAYOUT ---
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "map", "rowspan": 2}, {"type": "xy"}],
            [None,                          {"type": "xy"}]
        ],
        vertical_spacing=0.08,
        subplot_titles=("Hamiltonian Cycle",
                        "Energy Evolution", "Spin State Matrix")
    )

    # --- TRACE SETUP ---
    trace_idx = 0

    # 1. Background Energy Lines (Static)
    num_bg = min(50, energy_history.shape[1])
    for r in range(num_bg):
        fig.add_trace(go.Scatter(
            x=steps_x, y=energy_history[:, r], mode='lines',
            line=dict(color='rgba(150,150,150,0.1)', width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=2)
        trace_idx += 1

    # 2. Mean Energy (Static)
    fig.add_trace(go.Scatter(
        x=steps_x, y=mean_energy, mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Mean Energy'
    ), row=1, col=2)
    trace_idx += 1

    # --- INITIAL DYNAMIC TRACES (Rank 0) ---
    best = replica_data[0]
    # Determine initial color based on Rank 0 validity
    init_color = C_VALID if best['is_valid_bool'] else C_INVALID
    init_name = f"Route ({best['valid']})"

    # 3. Selected Energy (Dynamic ID: trace_idx)
    idx_energy = trace_idx
    fig.add_trace(go.Scatter(
        x=steps_x, y=energy_history[:, best['id']],
        mode='lines', line=dict(color=init_color, width=3),
        name='Selected Replica', showlegend=False  # Legend handled by map route
    ), row=1, col=2)
    trace_idx += 1

    # 4. Heatmap (Dynamic ID: trace_idx)
    idx_heatmap = trace_idx
    fig.add_trace(go.Heatmap(
        z=best['x_mat'],
        colorscale=[[0, 'white'], [1, init_color]],  # Dynamic Scale
        showscale=False, zmin=0, zmax=1,
        xgap=1, ygap=1,
        name="Permutation"
    ), row=2, col=2)
    trace_idx += 1

    # 5. Route Line (Dynamic ID: trace_idx)
    idx_route = trace_idx
    fig.add_trace(go.Scattermap(
        lat=[], lon=[], mode='lines',
        name=init_name,  # Initial Legend Name
        line=dict(width=4, color=init_color)
    ), row=1, col=1)
    trace_idx += 1

    # 6. Node Labels (Dynamic ID: trace_idx)
    idx_nodes = trace_idx
    fig.add_trace(go.Scattermap(
        lat=node_lats, lon=node_lons,
        mode='markers+text',
        textposition="top center",
        marker=dict(size=12, color='#636EFA'),
        textfont=dict(family="Arial", size=14, color="black"),
        name="Visit Order"
    ), row=1, col=1)
    trace_idx += 1

    # --- SLIDER LOGIC ---
    slider_steps = []

    # Target Indices: [Energy, Heatmap, Route, Nodes]
    target_indices = [idx_energy, idx_heatmap, idx_route, idx_nodes]

    for rank, data in enumerate(replica_data[:100]):
        # A. Determine Colors & Labels
        is_val = data['is_valid_bool']
        curr_color = C_VALID if is_val else C_INVALID
        curr_name = f"Route ({data['valid']})"

        # B. Calc Node Labels
        labels = [""] * L
        for step_i, node_id in enumerate(data['idx_seq']):
            real_idx = all_active_nodes.index(all_active_nodes[node_id])
            labels[real_idx] = str(step_i + 1)

        # C. Calc Route Lat/Lon
        rlats, rlons = [], []
        for u, v in zip(data['route'][:-1], data['route'][1:]):
            try:
                path = nx.shortest_path(G, u, v, weight='length')
                rlats.extend([G.nodes[n]['y'] for n in path] + [None])
                rlons.extend([G.nodes[n]['x'] for n in path] + [None])
            except:
                rlats.extend([G.nodes[u]['y'], G.nodes[v]['y'], None])
                rlons.extend([G.nodes[u]['x'], G.nodes[v]['x'], None])

        # D. Build Update Arrays for 'restyle'
        # We assume order: [Energy, Heatmap, Route, Nodes]

        # 1. Update Data (y, z, lat, lon, text)
        y_data = [energy_history[:, data['id']], None, None, None]
        z_data = [None, data['x_mat'], None, None]
        lat_data = [None, None, rlats, node_lats]
        lon_data = [None, None, rlons, node_lons]
        text_data = [None, None, None, labels]

        # 2. Update Visuals (Colors & Legend Names)
        # line.color: Updates Energy and Route
        line_color_data = [curr_color, None, curr_color, None]

        # colorscale: Updates Heatmap
        new_colorscale = [[0, 'white'], [1, curr_color]]
        colorscale_data = [None, new_colorscale, None, None]

        # name: Updates Legend for Route
        name_data = [None, None, curr_name, None]

        step = {
            "method": "restyle",
            "args": [{
                "y": y_data,
                "z": z_data,
                "lat": lat_data,
                "lon": lon_data,
                "text": text_data,
                # Style Updates
                "line.color": line_color_data,
                "colorscale": colorscale_data,
                "name": name_data
            }, target_indices],
            "label": f"{rank+1}"
        }
        slider_steps.append(step)

    # --- FINAL LAYOUT ---
    fig.update_layout(
        autosize=True, width=None, height=None,
        margin=dict(l=20, r=20, t=50, b=20),
        map_style="carto-positron",
        map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=13),
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Rank: "},
            "pad": {"t": 50},
            "steps": slider_steps
        }],
        title_text=f"TSP Optimization: {PLACE}",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )

    # Fix Heatmap Aspect
    fig.update_yaxes(scaleanchor='x2', scaleratio=1, constrain='domain',
                     showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=2)

    # Log Scale Energy
    fig.update_yaxes(title_text="Energy (Log)", type="log", row=1, col=2)

    return fig


fig = create_integrated_dashboard(
    G, system, all_active_nodes, W, full_energy_history)

# Save with a custom template to force full viewport height
fig.write_html(
    "examples/tsp_dashboard_fullscreen.html",
    include_plotlyjs='cdn',
    full_html=True,
    config={'responsive': True}  # This is the secret for dynamic resizing
)
print("Dashboard saved to examples/tsp_dashboard_fullscreen.html")
