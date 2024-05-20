from analytics_server.utils.constants import AccessLevels


def create_user(first_name, last_name, email, password, access_level):
    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": password,
        "access_level": access_level,
    }


admin_user = create_user(
    first_name="Jarno",
    last_name="Neuvonen",
    email="neuvonenjarno@gmail.com",
    password="password123",
    access_level=AccessLevels.ADMIN,
)
