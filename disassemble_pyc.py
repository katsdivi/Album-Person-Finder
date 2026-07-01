import dis
import marshal
import pathlib
import sys

def main():
    pyc_path = pathlib.Path("/Users/divyamkataria/Family Photo Finder/__pycache__/drive_client.cpython-312.pyc")
    if not pyc_path.exists():
        print(f"Error: {pyc_path} does not exist.")
        return

    data = pyc_path.read_bytes()
    # Skip the 16-byte header for Python 3.12
    code_obj = marshal.loads(data[16:])
    
    # Write disassembly to a file
    output_path = pathlib.Path("/Users/divyamkataria/Family Photo Finder/scratch_disassembly.txt")
    with open(output_path, "w") as f:
        sys.stdout = f
        dis.dis(code_obj)
        sys.stdout = sys.__stdout__
    print(f"Disassembled successfully to {output_path}")

if __name__ == "__main__":
    main()
