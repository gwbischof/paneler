# Custom Shapes

Add your own polyhedra as `.obj` files in this directory.

## Usage

1. Create or export a polyhedron as an `.obj` file
2. Save it in this directory (e.g., `myshape.obj`)
3. Use it: `uv run python main.py myshape`

## Creating OBJ Files

You can create `.obj` files using:
- **Blender** - Free 3D modeling software
- **MeshLab** - Free mesh processing tool
- **Code** - Generate programmatically and export

## OBJ Format Example

```obj
# Vertices
v 0.0 0.0 1.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v -1.0 0.0 0.0

# Faces (vertex indices, 1-based)
f 1 2 3
f 1 3 4
f 1 4 2
f 2 4 3
```

## Tips

- Keep your polyhedra centered at the origin
- Use consistent scale (vertices around unit distance from origin work well)
- Ensure faces are properly oriented (counter-clockwise when viewed from outside)
