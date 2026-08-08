import csv
import os

DATA_FILE = "isl_dataset.csv"

def count_samples():
    counts = {}
    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                label = row[-1].strip()
                counts[label] = counts.get(label, 0) + 1
    print("\nCurrent samples per letter:")
    for letter in sorted(counts.keys()):
        print(f"  {letter}: {counts[letter]}")
    print(f"  TOTAL: {sum(counts.values())}")

def delete_letter(letter):
    letter = letter.upper().strip()
    rows_kept = []
    rows_deleted = 0

    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[-1].strip().upper() == letter:
                rows_deleted += 1
            else:
                rows_kept.append(row)

    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_kept)

    print(f"✅ Deleted {rows_deleted} samples for letter '{letter}'")

def keep_only_n_samples(letter, n):
    letter = letter.upper().strip()
    rows_other = []
    rows_letter = []

    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[-1].strip().upper() == letter:
                rows_letter.append(row)
            else:
                rows_other.append(row)

    kept = rows_letter[:n]
    print(f"✅ Keeping only first {n} samples for '{letter}' (had {len(rows_letter)})")

    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_other)
        writer.writerows(kept)

# ── Run ──────────────────────────────────────────────────
print("=== Dataset Fix Tool ===")
count_samples()

print("\nOptions:")
print("  1. Delete ALL samples for a letter (then recollect)")
print("  2. Keep only first N samples for a letter")
print("  3. Delete LATEST N samples for a letter")
print("  4. Just view counts and exit")

choice = input("\nEnter choice (1/2/3/4): ").strip()

if choice == "1":
    letter = input("Which letter to delete completely? ").strip()
    delete_letter(letter)
    count_samples()

elif choice == "2":
    letter = input("Which letter? ").strip()
    n = int(input("Keep how many samples? ").strip())
    keep_only_n_samples(letter, n)
    count_samples()

elif choice == "3":
    letter = input("Which letter to delete LATEST N samples? ").strip()
    n = int(input("Delete how many latest samples? ").strip())
    letter = letter.upper().strip()
    rows_other = []
    rows_letter = []

    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[-1].strip().upper() == letter:
                rows_letter.append(row)
            else:
                rows_other.append(row)

    kept = rows_letter[:-n] if n < len(rows_letter) else []
    print(f"✅ Deleted last {n} samples for '{letter}' (had {len(rows_letter)}, now has {len(kept)})")

    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_other)
        writer.writerows(kept)

    count_samples()

elif choice == "4":
    print("Done.")
