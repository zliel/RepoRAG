// Sample Go file for testing
package main

import "fmt"

type User struct {
    ID    int
    Name  string
    Email string
}

type UserService struct {
    users map[int]User
}

func NewUserService() *UserService {
    return &UserService{
        users: make(map[int]User),
    }
}

func (s *UserService) AddUser(user User) {
    s.users[user.ID] = user
}

func (s *UserService) GetUser(id int) (User, bool) {
    user, ok := s.users[id]
    return user, ok
}

func main() {
    service := NewUserService()
    service.AddUser(User{ID: 1, Name: "Alice", Email: "alice@example.com"})
    user, ok := service.GetUser(1)
    if ok {
        fmt.Printf("Found user: %s\n", user.Name)
    }
}
