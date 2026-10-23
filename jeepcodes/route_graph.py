"""Accessible network diagrams without a plotting dependency."""

from colorsys import hsv_to_rgb
from base64 import b64encode
from html import escape
from functools import lru_cache
from math import hypot
from heapq import heappop, heappush


# Schematic stop arrangement from the supplied reference. Coordinates leave
# room for the existing node names, role labels, and connection annotations.
REFERENCE_POSITIONS = {
    "Alpha": (70, 262), "Bravo": (174, 347), "Charlie": (279, 347),
    "Delta": (383, 228), "Echo": (488, 352), "Foxtrot": (592, 259),
    "Golf": (695, 360), "Hotel": (798, 228), "India": (903, 350),
    "Juliet": (1007, 222), "Kilo": (1111, 355), "Lima": (1216, 337),
    "Mike": (1320, 329), "November": (1424, 106), "Oscar": (1530, 220),
    "Papa": (1635, 90), "Quebec": (1635, 220), "Romeo": (1768, 220),
}


@lru_cache(maxsize=32)
def network_positions(stops, connections):
    """Keep the reference arrangement identical across routes and scenarios."""
    positions = {stop: REFERENCE_POSITIONS[stop] for stop in stops if stop in REFERENCE_POSITIONS}
    # Retain visibility if a different validated network introduces new stops.
    for index, stop in enumerate(stop for stop in stops if stop not in positions):
        positions[stop] = (70 + index * 110, 430)
    return positions


def stop_colors(stops):
    """Give every stop a distinct color, shared by all displayed itineraries."""
    names = sorted(set(stops))
    colors = {}
    for index, name in enumerate(names):
        # Spread adjacent stops around the hue wheel rather than similar shades.
        hue = (index * 0.61803398875) % 1
        rgb = hsv_to_rgb(hue, 0.62, 0.88)
        colors[name] = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
    return colors


def boxes_overlap(first, second, gap=8):
    x, y, w, h = first
    a, b, c, d = second
    return x < a + c + gap and x + w + gap > a and y < b + d + gap and y + h + gap > b


def label_group(kind, box):
    x, y, w, h = box
    return (f'<g class="{kind}-label" data-x="{x}" data-y="{y}" '
            f'data-width="{w}" data-height="{h}">')


def segment_hits_box(start, end, box):
    """Liang-Barsky clipping, including the rectangle boundary."""
    x, y, w, h = box
    dx, dy = end[0] - start[0], end[1] - start[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, start[0] - x), (dx, x + w - start[0]),
                 (-dy, start[1] - y), (dy, y + h - start[1])):
        if p == 0:
            if q < 0:
                return False
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
        if low > high:
            return False
    return True


@lru_cache(maxsize=512)
def connection_points(origin, destination, all_positions):
    """Route both background and selected edges around nodes and their labels."""
    positions = dict(all_positions)
    if origin > destination:
        return tuple(reversed(connection_points(destination, origin, all_positions)))
    obstacles = []
    for stop, (x, y) in positions.items():
        if stop not in (origin, destination):
            obstacles.append((x - 30, y - 30, 60, 60))
        # Reserve the largest possible name/role block, even on other routes.
        label_width = max(len(stop) * 10 + 16, 93)
        left = max(12, x - label_width / 2)
        obstacles.append((left - 4, y + 26, label_width + 8, 72))

    def clear(a, b):
        return not any(segment_hits_box(a, b, box) for box in obstacles)

    start, end = positions[origin], positions[destination]
    if clear(start, end):
        return start, end
    # Visibility graph: obstacle corners provide a finite, deterministic set
    # of detours. No route may fall back to crossing an unrelated node.
    vertices = [start, end]
    for x, y, w, h in obstacles:
        for point in ((x-2, y-2), (x+w+2, y-2), (x-2, y+h+2), (x+w+2, y+h+2)):
            if point[0] >= 8 and point[1] >= 8 and clear(point, point):
                vertices.append(point)
    distances, previous = {0: 0.0}, {}
    queue = [(0.0, 0)]
    while queue:
        cost, index = heappop(queue)
        if cost != distances[index]:
            continue
        if index == 1:
            route = [end]
            while index != 0:
                index = previous[index]
                route.append(vertices[index])
            return tuple(reversed(route))
        for other, point in enumerate(vertices):
            if other == index or not clear(vertices[index], point):
                continue
            candidate = cost + hypot(point[0] - vertices[index][0], point[1] - vertices[index][1])
            if candidate < distances.get(other, float("inf")):
                distances[other] = candidate
                previous[other] = index
                heappush(queue, (candidate, other))
    raise ValueError(f"No clear graph connection between {origin} and {destination}")


def point_on_route(points, fraction):
    lengths = [hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:])]
    remaining = sum(lengths) * fraction
    for a, b, length in zip(points, points[1:], lengths):
        if remaining <= length:
            ux, uy = (b[0]-a[0])/length, (b[1]-a[1])/length
            return a[0] + remaining*ux, a[1] + remaining*uy, ux, uy
        remaining -= length
    a, b = points[-2:]
    length = lengths[-1]
    return b[0], b[1], (b[0]-a[0])/length, (b[1]-a[1])/length


def svg_route(points, kind, attributes=""):
    encoded = " ".join(f"{x},{y}" for x, y in points)
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        return (f'<line class="{kind}" data-points="{encoded}" x1="{x1}" y1="{y1}" '
                f'x2="{x2}" y2="{y2}" {attributes}>')
    commands = " L ".join(f"{x} {y}" for x, y in points)
    return (f'<path class="{kind}" data-points="{encoded}" d="M {commands}" '
            f'fill="none" stroke-linejoin="round" {attributes}>')


def edge_label_boxes(path, routes, obstacles, width):
    """Reserve conservative text bounds before drawing any connection labels."""
    occupied = list(obstacles)
    result = []
    for leg, route in zip(path.legs, routes):
        text_width = max(88, len(leg.segment.code) * 10 + 16,
                         len(f"{leg.costs.minutes:.2f} min") * 8 + 16)
        candidates = []
        for fraction in (0.5, 0.35, 0.65, 0.2, 0.8):
            x, y, ux, uy = point_on_route(route, fraction)
            if ux < 0 or (ux == 0 and uy < 0):
                ux, uy = -ux, -uy
            for offset in (0, -35, 35, -65, 65, -100, 100):
                cx = x - uy * offset
                cy = y + ux * offset - 14
                candidates.append((cx - text_width / 2, cy - 20, text_width, 40))
        box = next((candidate for candidate in candidates
                    if candidate[0] >= 12 and candidate[0] + candidate[2] <= width - 12
                    and candidate[1] >= 12 and candidate[1] + candidate[3] <= 500
                    and not any(boxes_overlap(candidate, other) for other in occupied)), None)
        # A free lower lane guarantees space even for unusually dense journeys.
        if box is None:
            y = 440
            while box is None:
                for x in range(20, int(width - text_width - 12), int(text_width + 16)):
                    candidate = (x, y, text_width, 40)
                    if not any(boxes_overlap(candidate, other) for other in occupied):
                        box = candidate
                        break
                y += 56
        result.append(box)
        occupied.append(box)
    return result


def route_graph_html(path, colors, network=None):
    """Show the full network with a highlighted itinerary in the reference arrangement."""
    width, height = 1850, 560
    stops = tuple(sorted(network.nodes if network is not None else colors))
    segments = network.segments if network is not None else (leg.segment for leg in path.legs)
    connections = tuple(sorted({tuple(sorted((edge.origin, edge.destination))) for edge in segments}))
    positions = network_positions(stops, connections)
    position_key = tuple(sorted(positions.items()))
    routes = [connection_points(leg.segment.origin, leg.segment.destination, position_key) for leg in path.legs]
    roles = {}
    role_lines = {}
    node_boxes = {}
    for stop, (x, y) in positions.items():
        index = path.stops.index(stop) if stop in path.stops else -1
        roles[stop] = ("" if index < 0 else
                       "Start / destination" if len(path.stops) == 1 else
                       "Start" if index == 0 else "Destination" if index == len(path.stops) - 1 else
                       "Transfer" if path.legs[index - 1].segment.code != path.legs[index].segment.code else "")
        role_lines[stop] = (["Start /", "destination"] if roles[stop] == "Start / destination"
                            else [roles[stop]] if roles[stop] else [])
        label_width = max([len(stop) * 10 + 16] + [len(line) * 7 + 16 for line in role_lines[stop]])
        label_left = max(12, min(width - label_width - 12, x - label_width / 2))
        node_boxes[stop] = (label_left, y + 30, label_width, 24 + 20 * len(role_lines[stop]))
    obstacles = list(node_boxes.values()) + [(x - 28, y - 28, 56, 56) for x, y in positions.values()]
    edge_boxes = edge_label_boxes(path, routes, obstacles, width)
    height = max(height, max((box[1] + box[3] + 60 for box in edge_boxes), default=0))
    description = " → ".join(path.stops)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(description, quote=True)}" '
        'style="display:block;max-width:none;font-family:Arial,sans-serif;color:#243746">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<title>Route network: {escape(description)}</title>',
        '<desc>All network stops follow the reference schematic arrangement. Orange arrows highlight the itinerary. '
        'Connections show the jeepney code and leg travel time. Spacing is schematic.</desc>',
    ]
    # Combine parallel services into one background connection per stop pair.
    if network is not None:
        pairs = {}
        for segment in network.segments:
            key = tuple(sorted((segment.origin, segment.destination)))
            pairs.setdefault(key, []).append(segment)
        for (origin, destination), segments in sorted(pairs.items()):
            x1, y1 = positions[origin]
            x2, y2 = positions[destination]
            available = any(segment.status == "OPEN" for segment in segments)
            stroke = "#b8c2cc" if available else "#cf6770"
            dash = "" if available else ' stroke-dasharray="6 5"'
            services = ", ".join(sorted({f"{segment.code} ({segment.status})" for segment in segments}))
            route = connection_points(origin, destination, position_key)
            parts.append(svg_route(route, "network-edge",
                                   f'stroke="{stroke}" stroke-width="1.5" opacity="0.55"{dash}') +
                         f'<title>{escape(origin)} to {escape(destination)}: {escape(services)}</title>' +
                         ('</line>' if len(route) == 2 else '</path>'))
    labels, leaders = [], []
    for leg, box, route in zip(path.legs, edge_boxes, routes):
        start_x, start_y, first_ux, first_uy = point_on_route(route, 0)
        end_x, end_y, ux, uy = point_on_route(route, 1)
        start_x, start_y = start_x + 25 * first_ux, start_y + 25 * first_uy
        end_x, end_y = end_x - 28 * ux, end_y - 28 * uy
        visible_route = ((start_x, start_y), *route[1:-1], (end_x, end_y))
        mid_x, mid_y, _, _ = point_on_route(route, 0.5)
        parts.extend([
            svg_route(visible_route, "selected-edge", 'stroke="#e88a16" stroke-width="4"') +
            ('</line>' if len(visible_route) == 2 else '</path>'),
            f'<polygon points="{end_x},{end_y} '
            f'{end_x-11*ux-6*uy},{end_y-11*uy+6*ux} '
            f'{end_x-11*ux+6*uy},{end_y-11*uy-6*ux}" fill="#e88a16"/>',
        ])
        label_x, label_y, label_width, label_height = box
        cx, cy = label_x + label_width / 2, label_y + label_height / 2
        if (hypot(cx - mid_x, cy - mid_y) > 55
                and not any(segment_hits_box((mid_x, mid_y), (cx, cy), obstacle)
                            for obstacle in obstacles + [other for other in edge_boxes if other != box])):
            leaders.append(f'<line class="label-guide" x1="{mid_x}" y1="{mid_y}" x2="{cx}" y2="{cy}" '
                          'stroke="#e88a16" stroke-width="1" stroke-dasharray="3 3"/>')
        labels.extend([
            label_group("edge", box),
            f'<rect x="{label_x}" y="{label_y}" width="{label_width}" height="{label_height}" rx="5" fill="white" opacity="0.97"/>',
            f'<text x="{cx}" y="{label_y + 16}" text-anchor="middle" fill="currentColor" '
            f'font-size="14" font-weight="600">{escape(leg.segment.code)}</text>',
            f'<text x="{cx}" y="{label_y + 32}" text-anchor="middle" fill="currentColor" '
            f'font-size="12">{leg.costs.minutes:.2f} min</text>',
            '</g>',
        ])
    parts.extend(leaders)
    parts.extend(labels)
    for stop, (x, y) in positions.items():
        index = path.stops.index(stop) if stop in path.stops else -1
        role = roles[stop]
        text_x = node_boxes[stop][0] + node_boxes[stop][2] / 2
        parts.extend([
            f'<circle cx="{x}" cy="{y}" r="23" fill="{colors[stop]}" stroke="currentColor" stroke-width="{3 if index >= 0 else 1}">'
            f'<title>{escape(stop)}{": " + role if role else ""}</title></circle>',
            label_group("node", node_boxes[stop]),
            f'<text x="{text_x}" y="{y + 47}" text-anchor="middle" fill="currentColor" '
            f'font-size="14" font-weight="600">{escape(stop)}</text>',
            ''.join(f'<text x="{text_x}" y="{y + 67 + index * 14}" text-anchor="middle" '
                    f'fill="currentColor" font-size="11">{escape(line)}</text>'
                    for index, line in enumerate(role_lines[stop])),
            '</g>',
        ])
    parts.append(f'<text x="24" y="{height - 20}" fill="#243746" font-size="13">'
                 'Orange arrows: selected route | Gray: other connections | Dashed red: closed connection</text>')
    parts.append('</svg>')
    # st.html's HTML-only sanitizer removes inline SVG. An image data URI keeps
    # the SVG intact while retaining the surrounding horizontal scroll region.
    encoded = b64encode("".join(parts).encode("utf-8")).decode("ascii")
    return (
        '<div style="overflow-x:auto; padding:8px 0" tabindex="0" '
        'role="region" aria-label="Route network graph; scroll horizontally for longer routes">'
        f'<img src="data:image/svg+xml;base64,{encoded}" '
        f'alt="Route network: {escape(description, quote=True)}" width="{width}" height="{height}" '
        'style="display:block;max-width:none"/></div>'
    )
