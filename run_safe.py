#!/usr/bin/env python3
import os
import sys
import traceback

# Set these environment variables to avoid MKL issues before importing NumPy
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'
os.environ['MKL_DISABLE'] = 'YES'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

print("Starting program with MKL workarounds...")

try:
    from run_with_features import main
    print("Successfully imported main function")
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import method...")
    try:
        # Execute the script directly
        exec(open("run_with_features.py").read())
    except Exception as e:
        print(f"Error executing run_with_features.py: {e}")
        print(traceback.format_exc())
except Exception as e:
    print(f"Error running main function: {e}")
    print(traceback.format_exc())

print("Program execution complete.") 