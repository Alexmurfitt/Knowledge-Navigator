import requests

def is_valid_package(package_line):
    if "==" in package_line:
        package = package_line.split("==")[0].strip()
    else:
        return False
    response = requests.get(f"https://pypi.org/pypi/{package}/json")
    return response.status_code == 200

input_file = "requirements.txt"
output_file = "requirements_clean.txt"

with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

valid_lines = []
invalid_lines = []

for line in lines:
    if is_valid_package(line):
        valid_lines.append(line)
    else:
        invalid_lines.append(line)

with open(output_file, "w") as f:
    f.write("\n".join(valid_lines))

print("✅ requirements_clean.txt creado.")
if invalid_lines:
    print("⚠️ Paquetes eliminados por ser inválidos o no disponibles en PyPI:")
    for line in invalid_lines:
        print("  -", line)
else:
    print("✅ Todos los paquetes eran válidos.")
