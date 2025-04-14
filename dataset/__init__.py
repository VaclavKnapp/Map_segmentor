import os
os.makedirs('dataset', exist_ok=True)
with open('dataset/__init__.py', 'w') as f:
    f.write('')
