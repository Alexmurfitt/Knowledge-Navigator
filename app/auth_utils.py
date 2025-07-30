from passlib.context import CryptContext
from app.database import fake_users_db


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
print(pwd_context.hash("admin123"))



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def get_user(username: str):
    return fake_users_db.get(username)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user:
        return False
    return pwd_context.verify(password, user["hashed_password"])

def register_user(username: str, password: str):
    if username in fake_users_db:
        print("❌ Usuario ya existe.")
        return False
    fake_users_db[username] = {
        "username": username,
        "hashed_password": "$2b$12$4TMLEwplnwkIWIMemmKyQebdPsytXvXPNdsJwJVStupgvDXDJyEtW"#hash_password(password),
    }
    print("✅ Usuario registrado correctamente.")
    return True

def login_flow():
    username = input("👤 Usuario: ")
    password = input("🔑 Contraseña: ")

    if authenticate_user(username, password):
        print("✅ Inicio de sesión exitoso. ¡Bienvenido,", username + "!")
    else:
        print("❌ Usuario o contraseña incorrectos.")

def register_flow():
    username = input("🆕 Nuevo usuario: ")
    password = input("🔐 Nueva contraseña: ")
    register_user(username, password)

# Menú principal
def main():
    while True:
        print("\n--- MENÚ ---")
        print("1. Iniciar sesión")
        print("2. Registrarse")
        print("3. Salir")
        choice = input("Elige una opción: ")

        if choice == "1":
            login_flow()
        elif choice == "2":
            register_flow()
        elif choice == "3":
            print("👋 Saliendo...")
            break
        else:
            print("❌ Opción no válida.")

if __name__ == "__main__":
    main()
