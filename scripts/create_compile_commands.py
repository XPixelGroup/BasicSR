import os


def create_compile_commands(path='build'):
    path = os.path.realpath(path)
    os.makedirs(path, exist_ok=True)
    ninja_files = []
    for root, dirs, files in os.walk('./'):
        for file in files:
            if file == "build.ninja":
                ninja_files.append(os.path.join(root, file))
    assert len(ninja_files) == 1, "The number of 'build.ninja' file must be 1."

    real_path = os.path.realpath(ninja_files[0])

    output_path = os.path.join(path, "compile_commands.json")
    result = os.popen(f"ninja -f {real_path} -t compdb")
    with open(output_path, 'w+') as f:
        f.write(result.read())
    print(f"Saved compile_commands.json at {os.path.relpath(output_path)}")
