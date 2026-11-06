# Verify syntax of Colab script template
import ast

code_to_check = """
import sys, os, json, time, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
"""
ast.parse(code_to_check)
print("Syntax check passed.")
