"""
LiDAR Point Cloud Visualizer
Converts .laz files to .obj format for 3D visualization.
Supports single files or multiple files stacked vertically.
"""

import argparse
import numpy as np
import laspy
from pathlib import Path
import glob


def load_laz_file(filepath):
    """Load a LAZ file and return point cloud as numpy array."""
    try:
        import lazrs  # Ensure LAZ backend is available
        las = laspy.read(filepath)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Get intensity if available
        intensity = None
        if hasattr(las, 'intensity'):
            intensity = las.intensity
        
        return points, intensity
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def points_to_obj(points, output_file, colors=None, point_size=1.0):
    """
    Convert point cloud to OBJ file format.
    
    Args:
        points: numpy array of shape (N, 3) with x, y, z coordinates
        output_file: path to output .obj file
        colors: optional numpy array of shape (N, 3) with RGB values (0-1)
        point_size: size multiplier for points
    """
    output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Point cloud exported from LAZ file\n")
        f.write(f"# Points: {len(points)}\n\n")
        
        # Write vertices
        if colors is not None:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.3f} {g:.3f} {b:.3f}\n")
        else:
            for x, y, z in points:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        
        # OBJ point format (vertices as individual points)
        f.write("\n# Points\n")
        for i in range(1, len(points) + 1):
            f.write(f"p {i}\n")
    
    print(f"✓ Saved {len(points)} points to {output_file}")


def intensity_to_color(intensity, colormap='gray'):
    """
    Convert intensity values to RGB colors.
    
    Args:
        intensity: numpy array of intensity values
        colormap: 'gray', 'hot', 'viridis', 'jet'
    
    Returns:
        numpy array of shape (N, 3) with RGB values (0-1)
    """
    if intensity is None:
        return None
    
    # Normalize intensity to 0-1
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    
    colors = np.zeros((len(intensity), 3))
    
    if colormap == 'gray':
        colors[:, 0] = intensity_norm
        colors[:, 1] = intensity_norm
        colors[:, 2] = intensity_norm
    
    elif colormap == 'hot':
        # Hot colormap: black -> red -> yellow -> white
        colors[:, 0] = np.minimum(1.0, intensity_norm * 3)
        colors[:, 1] = np.maximum(0, np.minimum(1.0, intensity_norm * 3 - 1))
        colors[:, 2] = np.maximum(0, intensity_norm * 3 - 2)
    
    elif colormap == 'viridis':
        # Simplified viridis: purple -> blue -> green -> yellow
        t = intensity_norm
        colors[:, 0] = np.sqrt(t)  # Red increases with sqrt
        colors[:, 1] = t * t        # Green increases quadratically
        colors[:, 2] = 1.0 - t      # Blue decreases
    
    elif colormap == 'jet':
        # Jet colormap: blue -> cyan -> green -> yellow -> red
        t = intensity_norm * 4
        colors[:, 0] = np.minimum(1.0, np.maximum(0, np.minimum(t - 1.5, 4.5 - t)))
        colors[:, 1] = np.minimum(1.0, np.maximum(0, np.minimum(t - 0.5, 3.5 - t)))
        colors[:, 2] = np.minimum(1.0, np.maximum(0, np.minimum(t + 0.5, 2.5 - t)))
    
    return colors


def height_to_color(points, colormap='rainbow'):
    """
    Color points based on their Z (height) coordinate.
    
    Args:
        points: numpy array of shape (N, 3)
        colormap: 'rainbow', 'terrain', 'ocean'
    
    Returns:
        numpy array of shape (N, 3) with RGB values (0-1)
    """
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    colors = np.zeros((len(points), 3))
    
    if colormap == 'rainbow':
        # Rainbow: blue (low) -> green -> yellow -> red (high)
        t = z_norm * 5
        colors[:, 0] = np.minimum(1.0, np.maximum(0, np.minimum(t - 1, 4 - t)))  # Red
        colors[:, 1] = np.minimum(1.0, np.maximum(0, np.minimum(t, 3 - t)))       # Green
        colors[:, 2] = np.minimum(1.0, np.maximum(0, 2 - t))                       # Blue
    
    elif colormap == 'terrain':
        # Terrain: blue (low) -> green -> brown -> white (high)
        t = z_norm
        colors[:, 0] = 0.5 + 0.5 * t       # Red increases
        colors[:, 1] = 0.3 + 0.4 * (1-t)   # Green peaks in middle
        colors[:, 2] = 0.8 * (1 - t)       # Blue at bottom
    
    elif colormap == 'ocean':
        # Ocean: dark blue -> light blue
        colors[:, 0] = 0.2 * z_norm
        colors[:, 1] = 0.4 + 0.4 * z_norm
        colors[:, 2] = 0.6 + 0.4 * z_norm
    
    return colors


def process_single_file(laz_file, output_file, color_by='height', colormap='rainbow', subsample=None):
    """Process a single LAZ file."""
    print(f"Loading {laz_file}...")
    points, intensity = load_laz_file(laz_file)
    
    if points is None:
        return False
    
    print(f"  Points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Subsample if requested
    if subsample and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points = points[indices]
        if intensity is not None:
            intensity = intensity[indices]
        print(f"  Subsampled to {len(points)} points")
    
    # Generate colors
    colors = None
    if color_by == 'intensity' and intensity is not None:
        colors = intensity_to_color(intensity, colormap)
    elif color_by == 'height':
        colors = height_to_color(points, colormap)
    elif color_by == 'none':
        colors = None
    
    # Save to OBJ
    points_to_obj(points, output_file, colors)
    return True


def process_multiple_files(laz_files, output_file, z_spacing=10.0, color_by='height', 
                          colormap='rainbow', subsample=None):
    """
    Process multiple LAZ files and stack them vertically.
    
    Args:
        laz_files: list of LAZ file paths
        output_file: output OBJ file path
        z_spacing: vertical spacing between scans
        color_by: how to color points ('height', 'intensity', 'file', 'none')
        colormap: color scheme to use
        subsample: maximum points per file (None for no subsampling)
    """
    all_points = []
    all_colors = []
    
    print(f"Processing {len(laz_files)} files...")
    
    for i, laz_file in enumerate(laz_files):
        print(f"\n[{i+1}/{len(laz_files)}] Loading {Path(laz_file).name}...")
        points, intensity = load_laz_file(laz_file)
        
        if points is None:
            continue
        
        print(f"  Points: {len(points)}")
        
        # Subsample if requested
        if subsample and len(points) > subsample:
            indices = np.random.choice(len(points), subsample, replace=False)
            points = points[indices]
            if intensity is not None:
                intensity = intensity[indices]
            print(f"  Subsampled to {len(points)} points")
        
        # Offset Z coordinate
        points[:, 2] += i * z_spacing
        
        # Generate colors for this file
        if color_by == 'file':
            # Color by file index (create gradient across files)
            t = i / max(len(laz_files) - 1, 1)
            file_color = np.array([t, 1-t, 0.5])
            colors = np.tile(file_color, (len(points), 1))
        elif color_by == 'intensity' and intensity is not None:
            colors = intensity_to_color(intensity, colormap)
        elif color_by == 'height':
            colors = height_to_color(points, colormap)
        else:
            colors = None
        
        all_points.append(points)
        if colors is not None:
            all_colors.append(colors)
    
    if not all_points:
        print("❌ No valid point clouds loaded")
        return False
    
    # Concatenate all points
    all_points = np.vstack(all_points)
    if all_colors:
        all_colors = np.vstack(all_colors)
    else:
        all_colors = None
    
    print(f"\n{'='*60}")
    print(f"Combined point cloud:")
    print(f"  Total points: {len(all_points)}")
    print(f"  X range: [{all_points[:, 0].min():.2f}, {all_points[:, 0].max():.2f}]")
    print(f"  Y range: [{all_points[:, 1].min():.2f}, {all_points[:, 1].max():.2f}]")
    print(f"  Z range: [{all_points[:, 2].min():.2f}, {all_points[:, 2].max():.2f}]")
    print(f"{'='*60}\n")
    
    # Save to OBJ
    points_to_obj(all_points, output_file, all_colors)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAZ point cloud files to OBJ format for visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with height coloring
  python laz_to_obj.py scan.laz -o output.obj

  # Directory of files stacked vertically
  python laz_to_obj.py ./lidar/ -o stacked.obj --stack

  # Color by intensity with hot colormap
  python laz_to_obj.py scan.laz -o output.obj --color-by intensity --colormap hot

  # Subsample to 10k points
  python laz_to_obj.py scan.laz -o output.obj --subsample 10000

  # Stack first 10 files with custom spacing
  python laz_to_obj.py ./lidar/ -o stacked.obj --stack --max-files 10 --spacing 15
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Input LAZ file or directory containing LAZ files')
    parser.add_argument('-o', '--output', type=str, default='output.obj',
                       help='Output OBJ file path')
    parser.add_argument('--stack', action='store_true',
                       help='Stack multiple files vertically (for directories)')
    parser.add_argument('--spacing', type=float, default=10.0,
                       help='Vertical spacing between stacked scans (meters)')
    parser.add_argument('--color-by', choices=['height', 'intensity', 'file', 'none'],
                       default='height',
                       help='How to color points')
    parser.add_argument('--colormap', choices=['gray', 'hot', 'viridis', 'jet', 'rainbow', 'terrain', 'ocean'],
                       default='rainbow',
                       help='Color scheme')
    parser.add_argument('--subsample', type=int, default=None,
                       help='Maximum points per file (for performance)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input is file or directory
    if input_path.is_file():
        # Single file mode
        if not input_path.suffix == '.laz':
            print(f"❌ Error: Input file must be .laz format")
            return
        
        success = process_single_file(
            input_path,
            args.output,
            color_by=args.color_by,
            colormap=args.colormap,
            subsample=args.subsample
        )
        
        if success:
            print(f"\n✓ Conversion complete!")
            print(f"  Open {args.output} in a 3D viewer (MeshLab, Blender, etc.)")
    
    elif input_path.is_dir():
        # Directory mode
        laz_files = sorted(glob.glob(str(input_path / '*.laz')))
        
        if not laz_files:
            print(f"❌ No .laz files found in {input_path}")
            return
        
        # Limit number of files if requested
        if args.max_files:
            laz_files = laz_files[:args.max_files]
        
        if args.stack:
            # Stack mode
            success = process_multiple_files(
                laz_files,
                args.output,
                z_spacing=args.spacing,
                color_by=args.color_by,
                colormap=args.colormap,
                subsample=args.subsample
            )
        else:
            # Process first file only
            print(f"Processing first file from directory...")
            print(f"  (Use --stack to combine all files)")
            success = process_single_file(
                laz_files[0],
                args.output,
                color_by=args.color_by,
                colormap=args.colormap,
                subsample=args.subsample
            )
        
        if success:
            print(f"\n✓ Conversion complete!")
            print(f"  Open {args.output} in a 3D viewer (MeshLab, Blender, CloudCompare, etc.)")
    
    else:
        print(f"❌ Error: {input_path} not found")


if __name__ == "__main__":
    main()