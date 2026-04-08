// Sample TypeScript file for testing
interface User {
    id: number;
    name: string;
    email: string;
}

type UserCallback = (user: User) => void;

class UserService {
    private users: Map<number, User> = new Map();

    addUser(user: User): void {
        this.users.set(user.id, user);
    }

    getUser(id: number): User | undefined {
        return this.users.get(id);
    }

    forEach(callback: UserCallback): void {
        this.users.forEach(callback);
    }
}

type ApiResponse<T> = {
    data: T;
    status: number;
    message: string;
};

async function fetchUser(id: number): Promise<ApiResponse<User>> {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}

export { User, UserService, ApiResponse, fetchUser };
