filename = "/home/manish/Desktop/projects/video_Generation/video_generation_from_scratch/annotation/test_video_anno.jsonl"

with open(filename, "r") as f:
    lines = f.readlines()

# Remember Python lists are 0-indexed, so line 5099 is at index 5098
bad_line = lines[5098] 

print(f"--- CONTENT OF LINE 5099 ---")
print(bad_line)
print(f"---------------------------")

# If you see two JSON objects like {"a":1}{"b":2}, you need to add a newline between them.