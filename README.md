# hyperbolic_db: Enhanced Hyperbolic Vector Database

An enhanced vector database implemented in Go, designed for efficient storage and similarity search of high-dimensional vectors within a hyperbolic space. This system utilizes the **Poincaré ball model** to represent data hierarchies and employs a specialized **Locality-Sensitive Hashing (LSH)** index for fast approximate nearest neighbor (ANN) search. Data persistence is handled by an embedded key-value store, **bbolt**, and all functionality is exposed via a RESTful HTTP API.

## ✨ Key Features

* **Hyperbolic Geometry**: Uses the Poincaré ball to effectively capture and represent hierarchical and tree-like relationships in data.
* **Approximate Nearest Neighbor (ANN) Search**: An LSH index, adapted for hyperbolic space, provides fast and scalable similarity search.
* **Persistence**: Data is reliably stored on disk using bbolt, ensuring that the database state is preserved across restarts.
* **Thread-Safe**: The system is designed for concurrent access, allowing for multiple read and write operations without data corruption.
* **HTTP API**: A simple and easy-to-use RESTful API for adding vectors, searching for neighbors, and retrieving database statistics.

---

## 🚀 Getting Started

This guide will walk you through setting up and running the database from scratch.

### Step 1: Install Prerequisites

Before you begin, make sure you have the necessary tools installed on your system.

* **Go**: The application is built with Go. Download and install Go version 1.18 or higher from the [official Go website](https://go.dev/dl/).
* **Git**: You need Git to clone the repository. If you don't have it, you can download it from the [official Git website](https://git-scm.com/downloads).
* **Terminal**: You'll be using your system's command-line interface (Terminal on macOS/Linux, or Command Prompt/PowerShell on Windows).

### Step 2: Clone the Repository

Open your terminal and run the following command to download a copy of the project's source code.


    git clone https://github.com/memicalem387/hyperbolic_db.git

### Step 3: Install Dependencies and Build

Navigate into the project directory and use the Go module system to install the required packages and build the application

    cd hyperbolic_db

Initialize Go Modules: This command creates a go.mod file to manage your dependencies. If you already have one, you can skip this step.


    go mod init hyperbolic_db

Install Dependencies: This command downloads and installs all the necessary packages your project needs, like bbolt and uuid.


    go mod tidy

Build the Application: This command compiles the source code into a single executable file.

    go build -o hyperbolic_db . 


Step 4: Run the Server

Now that the application is built, you can start the server.
Bash

    ./hyperbolic_db

The server will start on http://localhost:8080. It will automatically load any existing data or create a new database with sample data if it's the first time you're running it.

### Usage

The database is accessed via its HTTP API. You can use a command-line tool like curl or any HTTP client to interact with the endpoints.

Add a Vector

Add a new vector with an optional ID. If no ID is provided, one will be generated.
```

    Endpoint: POST /add

    Request Body:
    JSON

{
    "id": "optional_item_id",
    "vector": [0.1, 0.2, 0.3, 0.4]
}
```
Example:

```
curl -X POST http://localhost:8080/add -H "Content-Type: application/json" -d '{"vector": [0.6, 0.3]}'
```

Search for Nearest Neighbors

Search for the k nearest neighbors to a query vector. The search uses ANN by default but can be configured for exact (brute-force) search.

```
    Endpoint: POST /search

    Request Body:
    JSON

{
    "vector": [0.8, 0.35],
    "k": 5,
    "exact": false
}

```
Example:

```
    curl -X POST http://localhost:8080/search -H "Content-Type: application/json" -d '{"vector": [0.7, 0.2], "k": 3}'
```
Get Database Statistics

Retrieve information about the database, including the total number of items, LSH buckets, and hash functions.



```
    Endpoint: GET /stats

```
Example:

```
curl http://localhost:8080/stats
```

###  Architecture Overview

The system architecture is composed of several key components that work together to provide efficient vector search:

- HTTP API Layer: Handles incoming requests and translates them into database operations.

- HyperbolicDB: The core database engine that manages in-memory data, the LSH index, and persistence.

- LSH Index: An in-memory index that uses a specialized hash function to group similar hyperbolic vectors, enabling fast ANN searches.

- bbolt Storage: The persistence layer that stores all vector data on disk, ensuring data durability.

- Mathematical Core: Contains the core logic for calculating the Poincaré distance and normalizing vectors for hyperbolic space.

