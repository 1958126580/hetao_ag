import os
import glob
from pathlib import Path

def generate_bundle():
    base_dir = Path("e:/codes/hetao_ag")
    output_file = base_dir / "hetao_ag_colab.py"
    
    # Files to include
    files_to_bundle = []
    
    # 1. Source files
    files_to_bundle.extend(sorted(list(base_dir.glob("hetao_ag/**/*.py"))))
    
    # 2. Tests
    files_to_bundle.extend(sorted(list(base_dir.glob("tests/*.py"))))
    
    # 3. Examples
    files_to_bundle.extend(sorted(list(base_dir.glob("examples/*.py"))))
    
    # 4. Docs (Optional, maybe as creating README)
    # files_to_bundle.extend(sorted(list(base_dir.glob("README.md"))))

    print(f"Bundling {len(files_to_bundle)} files...")
    
    # Header of the Colab script
    script_header = '''# -*- coding: utf-8 -*-
"""
hetao_ag: Smart Agriculture & Livestock Library (Colab All-in-One)
==================================================================

Generates the complete hetao_ag package, runs tests, and executes examples.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies in Colab"""
    print("=[ Installing Dependencies ]" + "="*40)
    packages = [
        "numpy", "pandas", "scipy", "matplotlib", "pyyaml",
        "rasterio", "geopandas", "shapely",  # Space module
        "pulp",  # Opt module
        "opencv-python-headless", "torch", "torchvision" # Livestock module (headless for server)
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("Dependencies installed.\\n")

def write_file(path, content):
    """Write content to file, creating directories as needed"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

# ============================================================================
# File Contents
# ============================================================================

FILES = {
'''

    script_footer = '''
}

def setup_package():
    """Write all files to disk"""
    print("=[ Setting up Package ]" + "="*45)
    base_path = Path.cwd()
    
    for rel_path, content in FILES.items():
        full_path = base_path / rel_path
        write_file(full_path, content)
        
    # Ensure current directory is in python path
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))
    print("Package installed locally.\\n")

def run_tests():
    """Run the test suite"""
    print("=[ Running Tests ]" + "="*50)
    import pytest
    try:
        # Run our specific test script as a module or calling its main
        # But since we have tests/test_all.py, let's run that using subprocess for isolation
        # or import it. subprocess is safer for environment variables.
        pass
    except ImportError:
        pass
        
    # Using the existing test script
    test_script = Path("tests/test_all.py")
    if test_script.exists():
        print(f"Executing {test_script}...")
        result = subprocess.call([sys.executable, str(test_script)])
        if result == 0:
            print(">> ALL TESTS PASSED <<")
        else:
            print(">> TESTS FAILED <<")
    else:
        print("Test script not found!")
    print("")

def run_examples():
    """Run usage examples"""
    print("=[ Running Examples ]" + "="*47)
    
    examples = [
        "examples/example_core.py",
        "examples/example_soil.py",
        "examples/example_water.py",
        "examples/example_crop.py",
        "examples/example_livestock.py",
        "examples/example_space.py",
        "examples/example_opt.py",
        "examples/demo.py"
    ]
    
    for ex in examples:
        p = Path(ex)
        if p.exists():
            print(f"\\nRunning {ex}...")
            print("-" * 60)
            subprocess.call([sys.executable, str(p)])
            print("-" * 60)

if __name__ == "__main__":
    install_dependencies()
    setup_package()
    
    # Re-import to ensure modules are found
    import importlib
    importlib.invalidate_caches()
    
    run_tests()
    run_examples()
    
    print("\\n=[ COMPLETE ]" + "="*55)
    print("The hetao_ag library is now ready to use in this Colab session.")
    print("Try importing modules:")
    print("  from hetao_ag.core import ...")
    print("  from hetao_ag.soil import ...")
'''

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(script_header)
        
        for file_path in files_to_bundle:
            try:
                # Calculate relative path from base_dir
                rel_path = file_path.relative_to(base_dir).as_posix()
                
                # Read content
                with open(file_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()
                
                # Escape functionality for python string (mainly triple quotes and backslashes)
                # We use repr() but sliced to remove surrounding quotes? No, raw string `r'''...'''` is best but triple quotes inside content break it.
                # Safer is to repr() the content and let python parse it.
                
                f_out.write(f'    "{rel_path}": {repr(content)},\n')
                
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                
        f_out.write(script_footer)
        
    print(f"Bundle generated at: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    generate_bundle()
