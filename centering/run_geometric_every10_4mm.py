# run_geometric_every10_4mm.py

from cine_iterator_geometric_4mm_every10 import process_every_10

folder = r"F:\spheres\4mm-borosilicate"

results = process_every_10(folder)

print("\n=== DONE (geometric centering run) ===")
for path_str, info in results.items():
    print(f"{path_str}: best_frame={info['best_frame']} "
          f"dark={info['best_dark_fraction']:.4f}")
