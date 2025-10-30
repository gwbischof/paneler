# Paneler

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/f1f7273d-a336-4e34-8937-b9e860f3839d" />


Generate 2D cutting patterns for footbags from 3D polyhedra.

## Usage

```bash
# Install
uv sync

# List available shapes
uv run python main.py --list

# Generate cutting patterns
uv run python main.py icosahedron

# With custom size and seam allowance
uv run python main.py icosahedron --radius 5.0 --seam-allowance 0.2 --output my_ball.pdf

# With 3D visualization
uv run python main.py icosahedron --save-viz shape.html
```

## Available Shapes

- `tetrahedron` - 4 faces
- `cube` - 6 faces
- `octahedron` - 8 faces
- `dodecahedron` - 12 faces
- `icosahedron` - 20 faces
- `truncated_icosahedron` - 32 faces (soccer ball)
