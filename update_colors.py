#!/usr/bin/env python3
"""Update colors in app.py to new Nordic blue theme"""

with open('app.py', 'r') as f:
    content = f.read()

replacements = {
    '#c9956c': '#339af0',  # caramel -> blue
    '#7a9bb8': '#868e96',  # sky -> gray
    '#c28585': '#ff6b6b',  # rose -> red
    '#7a9b7a': '#51cf66',  # sage -> green
    '#c9a95c': '#fcc419',  # honey -> yellow
    '#3d3024': '#212529',  # text dark
    '#6b5d4d': '#495057',  # text secondary
    '#f5f1eb': '#f1f3f5',  # bg
    'rgba(122,155,122,0.12)': 'rgba(81,207,102,0.1)',  # sage glow
    'rgba(120,100,80,0.06)': 'rgba(0,0,0,0.03)',  # bg surface
    'rgba(201,149,108,0.12)': 'rgba(51,154,240,0.1)',  # caramel glow
    'rgba(201,149,108,0.1)': 'rgba(51,154,240,0.08)',  # caramel glow light
    'rgba(201,149,108,0.2)': 'rgba(51,154,240,0.15)',  # caramel glow strong
    '#8b7d6b': '#868e96',  # muted text
    '#a89b8a': '#adb5bd',  # light text
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('app.py', 'w') as f:
    f.write(content)

print('Colors updated successfully!')
