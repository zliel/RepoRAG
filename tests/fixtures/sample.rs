// Sample Rust file for testing
use std::collections::HashMap;

struct User {
    id: u32,
    name: String,
    email: String,
}

struct UserService {
    users: HashMap<u32, User>,
}

impl UserService {
    fn new() -> Self {
        UserService {
            users: HashMap::new(),
        }
    }

    fn add_user(&mut self, user: User) {
        self.users.insert(user.id, user);
    }

    fn get_user(&self, id: u32) -> Option<&User> {
        self.users.get(&id)
    }

    fn for_each<F>(&self, mut callback: F)
    where
        F: FnMut(&User),
    {
        for user in self.users.values() {
            callback(user);
        }
    }
}

fn main() {
    let mut service = UserService::new();
    service.add_user(User {
        id: 1,
        name: String::from("Alice"),
        email: String::from("alice@example.com"),
    });
    if let Some(user) = service.get_user(1) {
        println!("Found user: {}", user.name);
    }
}
