input_file = "dataset_ids.txt"
output_file = "dataset_small.txt"

max_tokens = 200000   # ~ quelques Mo

count = 0

with open(input_file, "r") as inp, open(output_file, "w") as out:
    for token in inp.read().split():
        out.write(token + " ")
        count += 1
        if count >= max_tokens:
            break

print("mini dataset créé")