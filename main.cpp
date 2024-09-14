#include <tensorflow/c/c_api.h>

#include <chrono>
#include <ctime>
#include <fstream>  // File I/O
#include <iomanip>  // For formatting dates
#include <iostream>
#include <queue>
#include <regex>
#include <sstream>  // String stream for parsing
#include <string>
#include <vector>

struct Task {
    std::string description;
    int priority;  // Lower numbers mean higher priority
    std::chrono::system_clock::time_point
        deadline;  // Store deadline as a time_point

    // Operator overloading to sort tasks by priority
    bool operator<(const Task &other) const {
        return priority > other.priority;  // Higher priority comes first
    }

    // Convert the task to a string (for saving to a file)
    std::string serialize() const {
        std::time_t deadlineTime =
            std::chrono::system_clock::to_time_t(deadline);
        std::stringstream ss;
        ss << description << ";" << priority << ";"
           << std::put_time(std::localtime(&deadlineTime), "%Y-%m-%d %H:%M:%S")
           << "\n";
        return ss.str();
    }

    // Load task from a string (for reading from a file)
    static Task deserialize(const std::string &taskStr) {
        std::stringstream ss(taskStr);
        std::string token;
        Task task;

        std::getline(ss, task.description, ';');
        std::getline(ss, token, ';');
        task.priority = std::stoi(token);

        std::getline(ss, token, ';');
        std::tm tm = {};
        std::stringstream timeStream(token);
        timeStream >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

        // Convert std::tm to time_t assuming it's in GMT/UTC
        std::time_t time = std::mktime(&tm);  // mktime uses local time
        if (time != -1) {
            std::tm *gmt = std::gmtime(&time);  // Convert to GMT/UTC
            if (gmt) {
                time = std::mktime(gmt);  // Convert back to time_t in GMT/UTC
            }
        }
        task.deadline = std::chrono::system_clock::from_time_t(time);

        // // Use std::timegm to handle UTC time
        // std::time_t utc_time = std::timegm(&tm);
        // task.deadline =
        //     std::chrono::system_clock::from_time_t(std::mktime(&tm));

        return task;
    }
};

// Function to read a file into a TF_Buffer
TF_Buffer *readFileToBuffer(const std::string &filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file: " << filePath << std::endl;
        return nullptr;
    }

    TF_Buffer *tfBuffer = TF_NewBufferFromString(buffer.data(), size);
    return tfBuffer;
}

// Load the model
TF_Graph *loadModel(const std::string &modelPath, TF_Session *&session) {
    TF_Status *status = TF_NewStatus();
    TF_SessionOptions *sess_opts = TF_NewSessionOptions();
    TF_Graph *graph = TF_NewGraph();

    session = TF_NewSession(graph, sess_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error creating session: " << TF_Message(status)
                  << std::endl;
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(sess_opts);
        TF_DeleteGraph(graph);
        return nullptr;
    }

    TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
    TF_Buffer *graph_def = readFileToBuffer(modelPath);
    if (graph_def == nullptr) {
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(sess_opts);
        TF_DeleteGraph(graph);
        TF_DeleteImportGraphDefOptions(graph_opts);
        return nullptr;
    }

    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    TF_DeleteBuffer(graph_def);
    TF_DeleteImportGraphDefOptions(graph_opts);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error loading model: " << TF_Message(status) << std::endl;
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(sess_opts);
        TF_DeleteGraph(graph);
        return nullptr;
    }

    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(sess_opts);
    return graph;
}

// Example of cleanup (you should also clean up session and status)
void cleanup(TF_Session *session, TF_Graph *graph, TF_Status *status) {
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

// Predict task priority
int predictTaskPriority(const std::string &description, int hour) {
    TF_Session *session;
    TF_Graph *graph = loadModel("task_priority_model", session);

    // Prepare input tensor
    // For simplicity, assume the input tensor has been prepared and filled
    // properly
    // ...

    // Run the model
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor *> input_values;
    std::vector<TF_Output> outputs;
    std::vector<TF_Tensor *> output_values;

    TF_Status *status = TF_NewStatus();
    TF_SessionRun(session, nullptr, inputs.data(), input_values.data(),
                  inputs.size(), outputs.data(), output_values.data(),
                  outputs.size(), nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error during inference: " << TF_Message(status)
                  << std::endl;
    }

    // Extract the predicted priority
    int predicted_priority =
        *static_cast<int *>(TF_TensorData(output_values[0]));

    // Clean up
    TF_DeleteStatus(status);
    for (auto tensor : input_values) TF_DeleteTensor(tensor);
    for (auto tensor : output_values) TF_DeleteTensor(tensor);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);

    return predicted_priority;
}

// Save historical task data for machine learning
void saveTaskHistory(const Task &task) {
    std::ofstream file("task_history.csv", std::ios::app);

    if (file.is_open()) {
        // Save the task description, deadline, and priority as a CSV line
        std::time_t deadlineTime =
            std::chrono::system_clock::to_time_t(task.deadline);
        file << task.description << "," << deadlineTime << "," << task.priority
             << "\n";
        file.close();
    } else {
        std::cerr << "Failed to open task history file.\n";
    }
}

// Helper function to parse natural language commands
bool parseNaturalLanguageCommand(
    const std::string &input, std::string &description,
    std::chrono::system_clock::time_point &deadline) {
    std::regex timeInPattern(R"(in (\d+) (minutes?|hours?|days?))");
    std::regex atTimePattern(R"(at (\d{1,2}):(\d{2}) ?(AM|PM)?)");
    std::regex todayPattern(R"(today at (\d{1,2}):(\d{2}) ?(AM|PM)?)");
    std::regex tomorrowPattern(R"(tomorrow at (\d{1,2}):(\d{2}) ?(AM|PM)?)");

    std::smatch match;

    auto now = std::chrono::system_clock::now();

    // Extract the task description (everything before the time info)
    std::size_t timePosition = input.find(" at ");
    if (timePosition != std::string::npos) {
        description = input.substr(0, timePosition);
    } else {
        description = input;
    }

    // The words we want to remove ("tomorrow" or "today")
    std::string wordsToRemove[] = {"tomorrow", "today"};
    for (const std::string &wordToRemove : wordsToRemove) {
        // Find the position of the word ("tomorrow" or "today")
        size_t pos = description.find(wordToRemove);

        // If the word is found
        if (pos != std::string::npos) {
            // Erase the word from the string
            description.erase(pos, wordToRemove.length());

            // Optionally, remove any extra space left after erasing the word
            // If there is a space before or after the word, remove it
            if (pos < description.length() && description[pos] == ' ') {
                description.erase(pos, 1);  // Erase the space after the word
            } else if (pos > 0 && description[pos - 1] == ' ') {
                description.erase(pos - 1,
                                  1);  // Erase the space before the word
            }
        }
    }

    // Check for "in X minutes/hours/days"
    if (std::regex_search(input, match, timeInPattern)) {
        int amount = std::stoi(match[1].str());
        std::string unit = match[2].str();

        if (unit.find("minute") != std::string::npos) {
            deadline = now + std::chrono::minutes(amount);
        } else if (unit.find("hour") != std::string::npos) {
            deadline = now + std::chrono::hours(amount);
        } else if (unit.find("day") != std::string::npos) {
            deadline = now + std::chrono::hours(amount * 24);
        }
        return true;
    }

    // Check for "today at X:XX AM/PM"
    if (std::regex_search(input, match, todayPattern)) {
        int hour = std::stoi(match[1].str());
        int minute = std::stoi(match[2].str());
        std::string period = match[3].str();

        if (period == "PM" && hour < 12) {
            hour += 12;
        } else if (period == "AM" && hour == 12) {
            hour = 0;
        }

        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time_t);
        now_tm.tm_hour = hour;
        now_tm.tm_min = minute;
        now_tm.tm_sec = 0;

        deadline = std::chrono::system_clock::from_time_t(std::mktime(&now_tm));
        return true;
    }

    // Check for "tomorrow at X:XX AM/PM"
    if (std::regex_search(input, match, tomorrowPattern)) {
        int hour = std::stoi(match[1].str());
        int minute = std::stoi(match[2].str());
        std::string period = match[3].str();

        if (period == "PM" && hour < 12) {
            hour += 12;
        } else if (period == "AM" && hour == 12) {
            hour = 0;
        }

        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time_t);
        now_tm.tm_hour = hour;
        now_tm.tm_min = minute;
        now_tm.tm_sec = 0;

        deadline = std::chrono::system_clock::from_time_t(std::mktime(&now_tm));
        deadline +=
            std::chrono::hours(24);  // Add 24 hours to represent tomorrow
        return true;
    }

    // Check for "at X:XX AM/PM"
    if (std::regex_search(input, match, atTimePattern)) {
        int hour = std::stoi(match[1].str());
        int minute = std::stoi(match[2].str());
        std::string period = match[3].str();

        if (period == "PM" && hour < 12) {
            hour += 12;
        } else if (period == "AM" && hour == 12) {
            hour = 0;
        }

        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time_t);
        now_tm.tm_hour = hour;
        now_tm.tm_min = minute;
        now_tm.tm_sec = 0;

        deadline = std::chrono::system_clock::from_time_t(std::mktime(&now_tm));
        return true;
    }

    return false;
}

int extractHour(const std::chrono::system_clock::time_point &time_point) {
    // Convert time_point to std::time_t
    std::time_t time = std::chrono::system_clock::to_time_t(time_point);

    // Convert std::time_t to std::tm (local time)
    std::tm *local_time = std::localtime(&time);

    // Extract the hour
    int hour = local_time->tm_hour;

    return hour;
}

void addTaskNaturalLanguage(std::priority_queue<Task> &tasks) {
    Task newTask;
    std::string input;

    std::cin.ignore();
    std::cout << "Enter task description and deadline (e.g., 'Remind me to "
                 "call John at 5:20 PM'): ";
    std::getline(std::cin, input);

    std::chrono::system_clock::time_point deadline;

    if (parseNaturalLanguageCommand(input, newTask.description, deadline)) {
        newTask.deadline = deadline;
        newTask.priority = 1;  // Set default priority
        tasks.push(newTask);
        saveTaskHistory(newTask);
        std::cout << "Task added successfully.\n";
    } else {
        std::cout << "Could not understand the command. Please try again.\n";
    }
}

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

bool validateDeadlineFormat(const std::string &deadlineStr) {
    std::regex deadlinePattern(R"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})");
    return std::regex_match(deadlineStr, deadlinePattern);
}

void addTask(std::priority_queue<Task> &tasks) {
    Task newTask;
    std::cout << "Enter task description: ";
    std::cin.ignore();
    std::getline(std::cin, newTask.description);
    std::cout << "Enter task priority (lower number = higher priority): ";
    std::cin >> newTask.priority;

    std::string deadlineStr;
    bool validDeadline = false;

    // Loop until the user inters a valid deadline
    while (!validDeadline) {
        std::cout << "Enter task deadline (YYYY-MM-DD HH:MM:SS): ";
        std::cin.ignore();
        std::getline(std::cin, deadlineStr);

        if (validateDeadlineFormat(deadlineStr)) {
            validDeadline = true;
        } else {
            std::cout << "Invalid deadline format. Please enter in the format "
                         "YYYY-MM-DD HH:MM:SS.\n";
        }
    }

    std::tm tm = {};
    std::stringstream ss(deadlineStr);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    tm.tm_hour -= 1;
    newTask.deadline = std::chrono::system_clock::from_time_t(std::mktime(&tm));

    tasks.push(newTask);
    std::cout << "Task added successfully.\n";
}

void checkReminders(const std::priority_queue<Task> &tasks) {
    auto now = std::chrono::system_clock::now();
    auto threshold =
        std::chrono::seconds(3600);  // Set reminder for tasks due within 1 hour

    auto tempTasks = tasks;  // Copy the queue to avoid modifying it
    bool foundReminder = false;

    while (!tempTasks.empty()) {
        Task task = tempTasks.top();
        tempTasks.pop();

        auto timeLeft_ns = task.deadline - now;
        auto timeLeft_s =
            std::chrono::duration_cast<std::chrono::seconds>(timeLeft_ns);
        std::cout << "\n";
        if (timeLeft_s <= threshold && timeLeft_s > std::chrono::seconds(0)) {
            std::time_t deadlineTime =
                std::chrono::system_clock::to_time_t(task.deadline);
            std::cout << "Reminder: Task '" << task.description
                      << "' is due soon! Deadline: "
                      << std::ctime(&deadlineTime);
            foundReminder = true;
        }
    }

    if (!foundReminder) {
        std::cout << "No upcoming tasks within the next hour.\n";
    }
}

void showTasks(const std::priority_queue<Task> &tasks) {
    checkReminders(tasks);

    if (tasks.empty()) {
        std::cout << "No tasks available.\n";
        return;
    }

    std::cout << "\nTasks:" << std::endl;
    auto tempTasks = tasks;  // Make a copy to preserve the original queue
    while (!tempTasks.empty()) {
        Task task = tempTasks.top();
        tempTasks.pop();

        std::time_t deadlineTime =
            std::chrono::system_clock::to_time_t(task.deadline);
        std::cout << "Task: " << task.description
                  << " | Priority: " << task.priority
                  << " | Deadline: " << std::ctime(&deadlineTime);
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
                addTaskNaturalLanguage(tasks);
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
