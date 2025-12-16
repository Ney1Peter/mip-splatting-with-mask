#!/usr/bin/env python3
import struct
import sys
from pathlib import Path

def check_png_properties(png_path):
    p = Path(png_path)
    if not p.exists():
        print(f"File not found: {p}")
        return False

    try:
        with open(p, 'rb') as f:
            # Check PNG signature
            if f.read(8) != b'\x89PNG\r\n\x1a\n':
                print("Not a valid PNG file.")
                return False

            # Read first chunk (IHDR)
            while True:
                chunk_len = struct.unpack('>I', f.read(4))[0]
                chunk_type = f.read(4)
                if chunk_type == b'IHDR':
                    break
                # Skip other chunks until IHDR
                f.seek(chunk_len + 4, 1)  # +4 for CRC

            # Parse IHDR
            width = struct.unpack('>I', f.read(4))[0]
            height = struct.unpack('>I', f.read(4))[0]
            bit_depth = f.read(1)[0]
            color_type = f.read(1)[0]
            compression = f.read(1)[0]
            filter_method = f.read(1)[0]
            interlace = f.read(1)[0]

            print(f"PNG Info:")
            print(f"  Width: {width}")
            print(f"  Height: {height}")
            print(f"  Bit depth: {bit_depth} bits per channel")
            print(f"  Color type: {color_type}")
            print(f"  Interlace: {'Yes' if interlace == 1 else 'No'}")

            # Color type meaning:
            # 0 = Grayscale
            # 2 = Truecolor (RGB)
            # 3 = Indexed-color
            # 4 = Grayscale + alpha
            # 6 = Truecolor + alpha

            if color_type == 2:
                color_desc = "RGB"
            elif color_type == 6:
                color_desc = "RGBA"
            elif color_type == 0:
                color_desc = "Grayscale"
            elif color_type == 3:
                color_desc = "Indexed (palette)"
            else:
                color_desc = f"Unknown ({color_type})"

            print(f"  Color format: {color_desc}")

            # COLMAP compatibility check
            if bit_depth != 8:
                print(f"WARNING: Bit depth is {bit_depth}, but COLMAP requires 8-bit.")
                return False
            if interlace == 1:
                print(f"ERROR: Interlaced PNG! COLMAP cannot read this.")
                return False
            if color_type not in (2, 6):  # RGB or RGBA
                print(f"WARNING: Color type {color_type} may not be fully supported.")
            if color_type == 6:
                print(f"WARNING: RGBA detected. COLMAP may ignore alpha, but better to use RGB.")

            print(f"PNG appears COLMAP-compatible (based on header).")
            return True

    except Exception as e:
        print(f"Error reading PNG: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 check_png_header.py <image.png>")
        sys.exit(1)
    check_png_properties(sys.argv[1])