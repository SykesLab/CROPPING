# run_every10_4mm.py
#
# Entry point: process every 10th droplet in 4mm-borosilicate.

from cine_iterator_test_4mm_every10 import process_every_10

folder = r"F:\spheres\4mm-borosilicate"
csv_name = "cine_summary_4mm_every10.csv"

results = process_every_10(folder, csv_output=csv_name)

print("\n=== DONE ===")
for path_str, info in results.items():
    print(
        f"{path_str}: best_frame={info['best_frame']} "
        f"dark={info['best_dark_fraction']:.4f}"
    )
