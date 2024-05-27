# Read the big file
with open('cleaned_ipl-latest.txt', 'r') as f:
    data = f.read()

# Split the data based on "Date:"
sections = data.split("Date")

# Write each section to a new file
for i, section in enumerate(sections[1:], 1):  # Start from index 1 to skip the first empty section
    with open(f'part_file_{i}.txt', 'w') as f:
        f.write("Date:" + section.strip())
