#include <chrono>
#include <fstream>  // File I/O
#include <iomanip>  // For formatting dates
#include <iostream>
#include <queue>
#include <sstream>  // String stream for parsing
#include <string>

struct Task {
    std::string description;
    int priority;  // Lower numbers mean higher priority
    std::string deadline;

    // Operator overloading to sort tasks by priority
    bool operator<(const Task &other) const {
        return priority > other.priority;  // Higher priority comes first
    }

    // Convert the task to a string (for saving to a file)
    std::string serialize() const {
        return description + ";" + std::to_string(priority) + ";" + deadline +
               "\n";
    }

    // Load task from a string (for reading from a file)
    static Task deserialize(const std::string &taskStr) {
        std::stringstream ss(taskStr);
        std::string token;
        Task task;

        std::getline(ss, task.description, ';');
        std::getline(ss, token, ';');
        task.priority = std::stoi(token);
        std::getline(ss, task.deadline, ';');

        return task;
    }
};

void loadTasks(std::priority_queue<Task> &tasks) {
    std::ifstream inFile("tasks.txt");
    if (!inFile) {
        std::cerr << "Error opening file for reading. File may not exist.\n";
        return;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        if (!line.empty()) {
            Task task = Task::deserialize(line);
            tasks.push(task);
        }
    }

    inFile.close();
    std::cout << "Tasks loaded successfully.\n";
}

void saveTasks(const std::priority_queue<Task> &tasks) {
    std::ofstream outFile("tasks.txt");
    if (!outFile) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }

    // Make a copy of the queue to avoid destroying the original
    auto tempTasks = tasks;
    while (!tempTasks.empty()) {
        Task task = tempTasks.top();
        outFile << task.serialize();  // Write each task to the file
        tempTasks.pop();
    }

    outFile.close();
    std::cout << "Tasks saved successfully.\n";
}

void addTask(std::priority_queue<Task> &tasks) {
    Task newTask;
    std::cout << "Enter task description: ";
    std::cin.ignore();
    std::getline(std::cin, newTask.description);
    std::cout << "Enter task priority (lower number = higher priority): ";
    std::cin >> newTask.priority;
    std::cout << "Enter task deadline: ";
    std::cin.ignore();
    std::getline(std::cin, newTask.deadline);

    tasks.push(newTask);
    std::cout << "Task added successfully.\n";
}

void showTasks(const std::priority_queue<Task> &tasks) {
    if (tasks.empty()) {
        std::cout << "No tasks available.\n";
        return;
    }

    auto tempTasks = tasks;  // Make a copy to preserve the original queue
    while (!tempTasks.empty()) {
        Task task = tempTasks.top();
        std::cout << "Task: " << task.description
                  << ", Priority: " << task.priority
                  << ", Deadline: " << task.deadline << '\n';
        tempTasks.pop();
    }
}

void removeTask(std::priority_queue<Task> &tasks) {
    if (tasks.empty()) {
        std::cout << "No tasks to remove.\n";
        return;
    }

    tasks.pop();  // Remove the highest priority task
    std::cout << "Task removed successfully.\n";
}

int main() {
    std::priority_queue<Task> tasks;
    loadTasks(tasks);  // Load tasks from file at startup

    int choice;

    while (true) {
        std::cout << "\nTask Manager\n";
        std::cout << "1. Add Task\n";
        std::cout << "2. Show Tasks\n";
        std::cout << "3. Remove Highest Priority Task\n";
        std::cout << "4. Exit\n";
        std::cout << "Enter your choice: ";
        std::cin >> choice;

        switch (choice) {
            case 1:
                addTask(tasks);
                saveTasks(tasks);  // Save tasks after adding
                break;
            case 2:
                showTasks(tasks);
                break;
            case 3:
                removeTask(tasks);
                saveTasks(tasks);  // Save tasks after removing
                break;
            case 4:
                std::cout << "Exiting...\n";
                return 0;
            default:
                std::cout << "Invalid choice, please try again.\n";
        }
    }
}
