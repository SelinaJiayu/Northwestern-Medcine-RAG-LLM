# CancerRAG On Premise Deployment Guide

This guide provides step-by-step instructions to deploy the CancerRAG application on an on-premise server setup. It covers the setup of the repository, Docker builds, database initialization, and service launch to get the application running efficiently and securely.


## Step 1: Repository Setup

1. **Clone the Repository**  
   Begin by cloning the CancerRAG Git repository to the server:
   ```bash
   git clone https://github.com/ayush9818/CancerRAG
   cd CancerRAG
   git checkout on-premise-deployment
   ```

2. **Environment Configuration**  
    - Create an `.env` file in the root directory from env_template to manage environment variables. This will hold sensitive information like database credentials, API keys, and configuration settings.

        ```bash
        cp env_template .env 
        ```
    - Fill the required environment variables in .env file
        ```bash
        POSTGRES_USER=<POSTGRES_USERNAME>
        POSTGRES_PASSWORD=<POSTGRES_PASSWORD>
        POSTGRES_DB=<POSTGRES_DB_NAME>
        INIT_MODE=<DB_INIT_FLAG> 
        ```
    - Environment Variables description
        
        1. POSTGRES_USER : Username to connect to database. Can be set to any name. Eg. **nu_troy**
        2. POSTGRES_PASSWORD : Password to connect to database. Eg. **password**
        3. POSTGRES_DB : Name of the dabase. Eg. **cancer_rag_db**
        4. INIT_MODE : There are two steps to setup. First step is the database initialization step, in that case this variable is set to 1, otherwise set to 0 for conversational tasks.




## Step 2: Docker Builds

This section covers building the Docker images for both the backend and frontend services of the CancerRAG application.

1. **Build Backend Service**  
   - Navigate to the backend service directory and build the Docker image.
     ```bash
     cd backend
     docker build -f Dockerfile -t cancer_rag_backend .
     ```

2. **Build Frontend Service**  
   - Navigate to the frontend service directory and build the Docker image. 
     ```bash
     cd frontend
     docker build -f Dockerfile -t cancer_rag_frontend .
     ```

## Step 3: Database Initialization

**Note** : If you have already setup the things once, and want to setup again by reseting everything, make sure you run the following commands to reset properly
```bash
docker volume rm -f cancerrag_postgres_data
docker volume prune
```

To set up the initial database and schema, follow these steps:

1. **Enable Initialization Mode**  
   - Locate the `.env` file in the project root directory.
   - Find the `INIT_MODE` flag and set it to `1` to enable initialization mode:
     ```plaintext
     INIT_MODE=1
     ```
   - This flag ensures that the application runs the database setup process.

2. **Run the Initialization Command**  
   - From the project root directory, execute the following command to create the database and initialize the schema:
     ```bash
     docker compose -f docker-compose-init.yml up -d
     ```
   - This command launches the services defined in `docker-compose-init.yml`, which is specifically configured for initializing the database. The `-d` flag allows the services to run in the background.
   
3. **Verify Initialization**  
   - Check the logs to confirm the schema has been created successfully. You can view the logs with:
     ```bash
     docker logs <container_name>
     ```
   - Replace `<container_name>` with the name of the container responsible for initialization.

4. **Disable Initialization Mode**  
   - Once the database schema has been successfully initialized, return to the `.env` file and set the `INIT_MODE` flag back to `0` to prevent re-running the initialization on subsequent startups.

5. **Run the data ingestion script**

    In this step, you will load initial data into the database, preparing it for retrieval in the RAG pipeline. Follow these instructions to install dependencies and run the ingestion script with your data file.

    a. **Install Required Python Libraries**  
    Ensure you have the necessary Python libraries installed by running:

    ```bash
    pip3 install requests pandas
    ```

    b. **Run the Data Ingestion Script**  
    With the dependencies installed, use the following command to run the data ingestion script and load the initial data into the database:

    ```bash
    python3 scripts/data_ingestion_api.py \
        --data-path data/data_files/capstone_final_data_v1.csv \
        --api-base-url http://localhost:8000
    ```

    - **`--data-path`**: Specify the path to your initial data file (in CSV format) containing the questions and answers.
    - **`--api-base-url`**: Set the base URL for the API server. Make sure the server is running and accessible at this address (e.g., `http://localhost:8000`).

    This command will process the data in batches, sending each entry to the database through the API, enabling it to be ready for efficient retrieval in your RAG pipeline.

## Step 4: Launching Chatbot


With the data successfully loaded into the database, the next step is to launch the chatbot service. This service will use the preloaded data to provide responses within the RAG pipeline. Follow these instructions to configure and start the chatbot:

1. **Disable Initialization Mode**  
   - Open the `.env` file in the project root directory.
   - Set the `INIT_MODE` flag to `0` to prevent reinitialization of the database:
     ```plaintext
     INIT_MODE=0
     ```
   - This ensures that the chatbot service runs in normal mode without re-ingesting data.

2. **Start the Chatbot Service**  
   - From the project root directory, use the following command to launch the chatbot service in detached mode:
     ```bash
     docker compose -f docker-compose.yml up -d
     ```

   - The `-d` flag runs the service in the background, allowing the chatbot to operate independently.

3. **Verify Service Status**  
   - Check that the chatbot service is running by viewing active containers:
     ```bash
     docker ps
     ```
4. **Access the Chatbot**
    - Once the service is running, the chatbot will be accessible at http://localhost:8501.
    - Open this URL in your web browser to interact with the chatbot.

Once the chatbot service is up, it will be ready to interact with users, retrieving relevant information from the database and providing responses based on the data in the RAG pipeline.
