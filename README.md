# Paneler

Calculate panel shapes for footbags from 3D shapes.


<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/f1f7273d-a336-4e34-8937-b9e860f3839d" /> 
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/0c4e1415-36d8-49e7-a066-b36cf2c48a03" />

--- 

<img width="256" height="256" alt="Screenshot 2025-10-29 at 11 48 07 PM" src="https://github.com/user-attachments/assets/06326649-cb98-4d5a-9a86-3fe1613580b2" />

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

All shapes are loaded from `.obj` files in the `shapes/` directory.

### Included Shapes
- `tetrahedron` - 4 triangular faces
- `cube` - 6 square faces
- `octahedron` - 8 triangular faces
- `dodecahedron` - 12 pentagonal faces
- `icosahedron` - 20 triangular faces
- `pyramid` - 6 triangular faces (example)
- `cube-diag` - Cube with diagonals (example)
- `truncated_icosahedron` - 32 faces (classic soccer ball, procedurally generated)

### Add Your Own
Add `.obj` files to the `shapes/` directory - they'll automatically appear in `--list`. See `shapes/README.md` for details.
