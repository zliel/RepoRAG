"""Sample module for parser tests."""


class AuthService:
    """Handles auth."""

    def verify(self, token: str) -> bool:
        def inner():
            return True

        return bool(token) and inner()


def login(user: str) -> None:
    print(user)
