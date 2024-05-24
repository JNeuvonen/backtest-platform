from analytics_server.utils.constants import AccessLevels


def create_user(first_name, last_name, email, access_level):
    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "access_level": access_level,
    }


admin_user = create_user(
    first_name="Jarno",
    last_name="Neuvonen",
    email="neuvonenjarno@gmail.com",
    access_level=AccessLevels.ADMIN,
)
